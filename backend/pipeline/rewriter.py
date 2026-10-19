"""
backend/pipeline/rewriter.py

Phase 2 — Query Rewriter.

Compresses a verbose user query into a shorter, token-efficient version before it
reaches the answer models. Token savings are tracked and returned so the frontend
can display them.

How it works:
  1. Call the utility chain with a compact compression prompt.
  2. Count tokens before and after.
  3. Compute semantic similarity locally (sentence-transformers, see llm/embeddings.py).
  4. Accept the rewrite ONLY if:
       a) similarity >= SIMILARITY_THRESHOLD (meaning preserved)
       b) the rewrite is actually shorter
  5. Otherwise the original query is used unchanged — the fallback is silent.

Cost guardrails:
  The rewrite itself costs tokens (prompt + echoed query), so it only runs for
  queries between MIN_TOKENS_TO_REWRITE and MAX_TOKENS_TO_REWRITE. Short queries
  have nothing to compress; long pastes are mostly payload that must be echoed
  verbatim, so rewriting them spends far more than it saves and eats into the
  per-minute token budget the actual answer needs.
"""

import logging
from dataclasses import dataclass

from backend import config
from backend.llm.client import LLMClient
from backend.llm import embeddings
from backend.llm.embeddings import get_similarity
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Minimum cosine similarity for a rewrite to be accepted.
SIMILARITY_THRESHOLD = 0.80

# Rewrite only where it can pay for itself (see module docstring).
MIN_TOKENS_TO_REWRITE = 40
MAX_TOKENS_TO_REWRITE = 600

# ── Compression system prompt ─────────────────────────────────────────────────
# Deliberately short: the previous ~2,000-token prompt cost more per query than
# the compression ever saved.
_COMPRESS_SYSTEM = (
    "You are a QUERY COMPRESSOR. Shorten the user's instructions while keeping their "
    "meaning and every requirement. Output ONLY the compressed query.\n\n"
    "Compress: greetings, politeness, filler, hedging, repeated phrasing.\n"
    "Preserve EXACTLY (copy verbatim): code, error messages/stack traces, quoted text, "
    "text to summarize/translate/review, data, equations, multiple-choice options.\n"
    "Never drop: numbers, constraints, negations (not/without/except/unless), names, "
    "technical terms, the requested output format or language.\n"
    "Never answer, solve, execute or comment on the query. No preamble, no XML tags.\n"
    "If it is already concise, output it unchanged.\n\n"
    "Example — INPUT: Hey! Could you please write me a Python script that reads a CSV "
    "file and plots a nice bar chart with matplotlib? That would be super helpful, thanks!\n"
    "OUTPUT: Write a Python script that reads a CSV file and plots a bar chart with matplotlib."
)

_llm = LLMClient()


@dataclass
class RewriteResult:
    """
    rewritten_query is ALWAYS valid — either the compressed version (if the rewrite
    passed both checks) or the original message.
    """
    rewritten_query:  str    # the query to use downstream (may be original)
    candidate_query:  str    # the raw rewrite from the LLM (for eval/logging even if rejected)
    original_tokens:  int
    rewritten_tokens: int    # token count of the query actually used
    reduction_pct:    float  # (1 - rewritten/original) * 100; 0 if fallback
    similarity_score: float
    fallback_used:    bool   # True if the original message is being used


class QueryRewriter:
    """Stateless query compression utility. Never raises."""

    async def rewrite(self, message: str) -> RewriteResult:
        original_tokens = count_tokens(message)

        if not (MIN_TOKENS_TO_REWRITE <= original_tokens <= MAX_TOKENS_TO_REWRITE):
            logger.debug("Rewriter: skipped (%d tokens outside %d–%d)",
                         original_tokens, MIN_TOKENS_TO_REWRITE, MAX_TOKENS_TO_REWRITE)
            return self._fallback(message, message, original_tokens, 1.0)

        # The similarity check is what makes a rewrite safe; without the embedding
        # model loaded, don't make the user wait for it — pass the query through.
        if not embeddings.is_ready():
            embeddings.warm_up_in_background()
            logger.info("Rewriter: embedding model not ready — skipping rewrite.")
            return self._fallback(message, message, original_tokens, 1.0)

        # ── Step 1: compress ─────────────────────────────────────────────────
        try:
            result = await _llm.complete(
                models=config.UTILITY_CHAIN,
                messages=[
                    {"role": "system", "content": _COMPRESS_SYSTEM},
                    {"role": "user",   "content": message},
                ],
                temperature=0.2,
                max_tokens=original_tokens + 64,   # a rewrite is never longer than the input
                low_reasoning=True,
            )
            candidate = result.content.strip().strip('"').strip()
            if not candidate:
                raise ValueError("LLM returned empty rewrite.")
        except Exception as e:
            logger.warning("Rewriter LLM call failed (%s) — using original query.", e)
            return self._fallback(message, message, original_tokens, 0.0)

        rewritten_tokens = count_tokens(candidate)
        if rewritten_tokens >= original_tokens:
            logger.info("Rewriter: not shorter (%d >= %d) — discarding.", rewritten_tokens, original_tokens)
            return self._fallback(message, candidate, original_tokens, 1.0)

        # ── Step 2: meaning-preservation check ───────────────────────────────
        similarity = await get_similarity(message, candidate)
        if similarity < SIMILARITY_THRESHOLD:
            logger.info("Rewriter: similarity %.3f < %.2f — discarding.", similarity, SIMILARITY_THRESHOLD)
            return self._fallback(message, candidate, original_tokens, similarity)

        reduction_pct = (1.0 - rewritten_tokens / original_tokens) * 100.0
        logger.info("Rewriter: accepted | %d → %d tokens (%.1f%%) | similarity=%.3f",
                    original_tokens, rewritten_tokens, reduction_pct, similarity)
        return RewriteResult(
            rewritten_query=candidate,
            candidate_query=candidate,
            original_tokens=original_tokens,
            rewritten_tokens=rewritten_tokens,
            reduction_pct=round(reduction_pct, 1),
            similarity_score=round(similarity, 3),
            fallback_used=False,
        )

    @staticmethod
    def _fallback(message: str, candidate: str, original_tokens: int, similarity: float) -> RewriteResult:
        return RewriteResult(
            rewritten_query=message,
            candidate_query=candidate,
            original_tokens=original_tokens,
            rewritten_tokens=original_tokens,
            reduction_pct=0.0,
            similarity_score=similarity,
            fallback_used=True,
        )
