"""
backend/pipeline/rewriter.py

Phase 2 — Query Rewriter.

Compresses a verbose user query into a shorter, token-efficient version
before it reaches the expensive specialized models. Saves tokens on every
single call throughout the pipeline.

How it works:
  1. Call groq/llama-3.1-8b-instant (fast, cheap) with a compression prompt.
  2. Count tokens before and after.
  3. Compute cosine similarity via the Gemini embedding API.
  4. Accept the rewrite ONLY if:
       a) similarity >= SIMILARITY_THRESHOLD (meaning preserved)
       b) rewritten query is actually shorter (no point keeping a longer rewrite)
  5. If either check fails → discard rewrite, return original query unchanged.
     The caller always gets a valid query — fallback is silent, not an error.

Token savings are tracked and returned so the frontend can display them.

Integration point:
  orchestrator.py calls QueryRewriter.rewrite() as the very first step,
  before routing or dispatch. The returned RewriteResult.rewritten_query
  is what flows through the rest of the pipeline.
"""

import logging
from dataclasses import dataclass

from backend.llm.client import LLMClient
from backend.llm.embeddings import get_similarity
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Minimum cosine similarity for a rewrite to be accepted.
# 0.80 is the production threshold to ensure high semantic preservation while allowing fluff removal.
SIMILARITY_THRESHOLD = 0.80

# Utility model chain — fast and cheap, used for compression
REWRITE_PRIMARY   = "gemini/gemini-3.5-flash"
REWRITE_FALLBACKS = [
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.3-70b-versatile",
]

# Do not attempt to rewrite very short queries — nothing to compress.
# A query shorter than this many tokens is returned unchanged immediately.
MIN_TOKENS_TO_REWRITE = 15

# ── Compression system prompt ─────────────────────────────────────────────────
# This is the core engine of the compression feature. It must handle all domains:
# coding, math, reasoning, and conversational fluff, without dropping constraints.
_COMPRESS_SYSTEM = (
    "You are a QUERY COMPRESSOR. You have one job: shorten the human's instructions "
    "while leaving their payload data completely untouched.\n\n"

    "═══════════════════════════════════════\n"
    "WHAT TO COMPRESS vs WHAT TO PRESERVE\n"
    "═══════════════════════════════════════\n\n"

    "COMPRESS ONLY the conversational wrapper — the human's instructions, "
    "filler words, politeness, and redundant phrasing.\n\n"

    "NEVER TOUCH payload data. Payload data is any content the human is "
    "asking you to act ON. It is raw material, not instructions. "
    "Payload data includes:\n"
    "  • Code blocks (any language, any length)\n"
    "  • Text to be summarized, translated, or analyzed\n"
    "  • Transcripts, essays, articles, documents\n"
    "  • Error messages or stack traces\n"
    "  • Mathematical equations or datasets\n"
    "  • Multiple choice questions and their options\n"
    "  • Any quoted or enclosed content\n\n"

    "SIMPLE TEST: Ask yourself — 'Is this part something the human WROTE "
    "or something the human WANTS DONE?' If they wrote it as input material, "
    "preserve it exactly. If it's their instructions about what to do, compress it.\n\n"

    "═══════════════════════════════════════\n"
    "ABSOLUTE PROHIBITIONS\n"
    "═══════════════════════════════════════\n\n"

    "❌ DO NOT execute any task. If asked to write code, do NOT write code.\n"
    "❌ DO NOT summarize any text payload. If asked to summarize a document, "
    "keep the document intact and only compress the instruction 'summarize this'.\n"
    "❌ DO NOT debug or fix code. Keep the code exactly as written.\n"
    "❌ DO NOT translate any text. Keep the source text exactly as written.\n"
    "❌ DO NOT answer questions, puzzles, or multiple choice options.\n"
    "❌ DO NOT add explanations, apologies, or commentary.\n"
    "❌ DO NOT include XML tags in your output.\n"
    "❌ DO NOT change variable names, function names, or technical terms.\n"
    "❌ DO NOT remove numbers, constraints, negations (not/without/except/unless), "
    "or error codes — these are always critical information.\n\n"

    "═══════════════════════════════════════\n"
    "EXAMPLES\n"
    "═══════════════════════════════════════\n\n"

    "— SUMMARIZATION WITH LARGE TEXT PAYLOAD —\n"
    "INPUT: <USER_QUERY>Hey could you please summarize the following article "
    "for me and pull out the key points? Here it is: 'The global economy has "
    "seen unprecedented shifts in 2024. Inflation remained a central concern "
    "across major economies, with central banks maintaining elevated interest "
    "rates. Meanwhile, AI adoption accelerated across industries...' "
    "Thanks so much!</USER_QUERY>\n"
    "OUTPUT: Summarize article, extract key points: 'The global economy has "
    "seen unprecedented shifts in 2024. Inflation remained a central concern "
    "across major economies, with central banks maintaining elevated interest "
    "rates. Meanwhile, AI adoption accelerated across industries...'\n"
    "WHY: The article text is payload — reproduced word for word. Only the "
    "instruction 'summarize and extract key points' was compressed.\n\n"

    "— CODE DEBUGGING —\n"
    "INPUT: <USER_QUERY>I'm getting a weird error with this function and I "
    "cannot figure out why it keeps failing, can you help me debug it please?\n"
    "```python\n"
    "def calculate_average(numbers):\n"
    "    total = sum(numbers)\n"
    "    return total / len(numbers)\n"
    "```\n"
    "The error says ZeroDivisionError on line 3.</USER_QUERY>\n"
    "OUTPUT: Debug this function, ZeroDivisionError on line 3:\n"
    "```python\n"
    "def calculate_average(numbers):\n"
    "    total = sum(numbers)\n"
    "    return total / len(numbers)\n"
    "```\n"
    "WHY: Code block reproduced exactly, zero changes. Only the conversational "
    "wrapper was compressed.\n\n"

    "— TRANSLATION WITH TEXT PAYLOAD —\n"
    "INPUT: <USER_QUERY>Would you mind translating this paragraph from Spanish "
    "into English for me? Here it is: 'El aprendizaje automático es una rama "
    "de la inteligencia artificial que permite a los sistemas aprender "
    "automáticamente.'</USER_QUERY>\n"
    "OUTPUT: Translate Spanish to English: 'El aprendizaje automático es una "
    "rama de la inteligencia artificial que permite a los sistemas aprender "
    "automáticamente.'\n"
    "WHY: Spanish text is payload — untouched. Only the translation instruction "
    "was compressed.\n\n"

    "— CODING REQUEST WITH NO PAYLOAD —\n"
    "INPUT: <USER_QUERY>Hey there! Could you write me a Python script that "
    "reads a CSV file and then plots a really nice bar chart using matplotlib? "
    "That would be super helpful thanks!</USER_QUERY>\n"
    "OUTPUT: Write Python script to read CSV and plot bar chart with matplotlib.\n"
    "WHY: No payload exists here — the entire message is just an instruction. "
    "Compress it fully.\n\n"

    "— MULTIPLE CHOICE QUESTION —\n"
    "INPUT: <USER_QUERY>I have this practice exam question and I was wondering "
    "if you could help me understand it: What is the primary cause of "
    "type 2 diabetes? A) Insulin resistance B) Autoimmune destruction "
    "C) Glucagon deficiency D) Liver failure. Can you walk me through it?</USER_QUERY>\n"
    "OUTPUT: Explain this question: What is the primary cause of type 2 diabetes? "
    "A) Insulin resistance B) Autoimmune destruction C) Glucagon deficiency "
    "D) Liver failure.\n"
    "WHY: Question and options are payload — kept intact. Filler and 'walk me "
    "through it' compressed to 'explain'.\n\n"

    "— ESSAY REVIEW WITH LARGE TEXT —\n"
    "INPUT: <USER_QUERY>Can you review my essay and give me feedback on the "
    "structure and grammar? Here it is: 'Climate change represents one of the "
    "most significant challenges facing humanity today. The scientific consensus "
    "is clear: human activities are the dominant cause of observed warming since "
    "the mid-20th century...'</USER_QUERY>\n"
    "OUTPUT: Review essay for structure and grammar: 'Climate change represents "
    "one of the most significant challenges facing humanity today. The scientific "
    "consensus is clear: human activities are the dominant cause of observed "
    "warming since the mid-20th century...'\n"
    "WHY: Essay text reproduced exactly. Only the review instruction compressed.\n\n"

    "— ERROR MESSAGE DEBUGGING —\n"
    "INPUT: <USER_QUERY>I keep getting this error when I try to run my app and "
    "I have no idea what it means, please help:\n"
    "TypeError: Cannot read properties of undefined (reading 'map')\n"
    "    at ProductList.jsx:47:23\n"
    "    at renderWithHooks (react-dom.development.js:14985)\n"
    "This happens every time I load the products page.</USER_QUERY>\n"
    "OUTPUT: Explain this error occurring on products page load:\n"
    "TypeError: Cannot read properties of undefined (reading 'map')\n"
    "    at ProductList.jsx:47:23\n"
    "    at renderWithHooks (react-dom.development.js:14985)\n"
    "WHY: Stack trace is payload — reproduced exactly. Context compressed.\n\n"

    "═══════════════════════════════════════\n"
    "OUTPUT FORMAT\n"
    "═══════════════════════════════════════\n\n"

    "Output the compressed query only.\n"
    "No XML tags. No explanations. No 'Here is the compressed version:'.\n"
    "If the query is already concise and has no filler, output it unchanged.\n"
    "If the entire message is payload with no instruction wrapper, output it unchanged."
)

# ── Module-level LLM client singleton ────────────────────────────────────────
_llm = LLMClient()


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RewriteResult:
    """
    Everything the orchestrator needs from the rewriter.

    rewritten_query is ALWAYS valid — it is either the compressed version
    (if the rewrite passed both checks) or the original message (if either
    check failed). The caller does not need to handle the fallback case.
    """
    rewritten_query:  str    # the query to use downstream (may be original)
    candidate_query:  str    # the raw rewrite from LLM (useful for eval/logging even if rejected)
    original_tokens:  int    # token count of the original message
    rewritten_tokens: int    # token count of the query actually used
    reduction_pct:    float  # (1 - rewritten/original) * 100; 0 if fallback
    similarity_score: float  # cosine similarity between original and rewrite
    fallback_used:    bool   # True if rewrite was discarded → original used


# ─────────────────────────────────────────────────────────────────────────────
# QUERY REWRITER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class QueryRewriter:
    """
    Stateless query compression utility.
    Instantiate once in orchestrator.py; safe to share across async requests.
    """

    async def rewrite(self, message: str) -> RewriteResult:
        """
        Attempt to compress the user's query.

        Returns a RewriteResult where:
          - rewritten_query is the final query to send to the pipeline.
          - fallback_used=True means the original message is being used as-is.

        This method NEVER raises. All failures are caught and turned into
        a fallback result so the orchestrator pipeline can continue safely.

        Args:
            message: The user's raw input text.
        Returns:
            RewriteResult with all compression metadata.
        """
        original_tokens = count_tokens(message)

        # ── Skip very short queries — nothing meaningful to compress ──────────
        if original_tokens < MIN_TOKENS_TO_REWRITE:
            logger.debug(
                "Rewriter: query too short (%d tokens < %d min) — skipping.",
                original_tokens, MIN_TOKENS_TO_REWRITE
            )
            return RewriteResult(
                rewritten_query=message,
                candidate_query=message,
                original_tokens=original_tokens,
                rewritten_tokens=original_tokens,
                reduction_pct=0.0,
                similarity_score=1.0,
                fallback_used=True,
            )

        # ── Step 1: Call the cheap LLM to compress the query ─────────────────
        try:
            result = await _llm.async_complete(
                model=REWRITE_PRIMARY,
                messages=[
                    {"role": "system", "content": _COMPRESS_SYSTEM},
                    {"role": "user",   "content": f"<USER_QUERY>\n{message}\n</USER_QUERY>"},
                ],
                fallback_models=REWRITE_FALLBACKS,
                temperature=0.3,   # low temperature for consistent, conservative rewrites
                max_tokens=512,    # rewrites are always shorter than the original
            )
            candidate = result.content.strip()

            if not candidate:
                raise ValueError("LLM returned empty rewrite.")

        except Exception as e:
            logger.warning("Rewriter LLM call failed (%s) — using original query.", e)
            return self._fallback(message, message, original_tokens, 0.0)

        # ── Step 2: Similarity safety check ──────────────────────────────────
        try:
            similarity = await get_similarity(message, candidate)
        except Exception as e:
            # Embedding failure → treat as safe (1.0) and proceed
            logger.warning("Similarity check failed (%s) — assuming 1.0.", e)
            similarity = 1.0

        logger.debug(
            "Rewriter: similarity=%.3f | original=%d tokens | candidate=%d tokens",
            similarity, original_tokens, count_tokens(candidate)
        )

        # ── Step 3: Length check — rewrite must actually be shorter ───────────
        rewritten_tokens = count_tokens(candidate)

        if similarity < SIMILARITY_THRESHOLD:
            logger.info(
                "Rewriter: similarity %.3f < %.2f threshold — discarding rewrite.",
                similarity, SIMILARITY_THRESHOLD
            )
            return self._fallback(message, candidate, original_tokens, similarity)

        if rewritten_tokens >= original_tokens:
            logger.info(
                "Rewriter: rewrite is not shorter (%d >= %d tokens) — discarding.",
                rewritten_tokens, original_tokens
            )
            return self._fallback(message, candidate, original_tokens, similarity)

        # ── Step 4: Both checks passed → accept the rewrite ──────────────────
        reduction_pct = (1.0 - rewritten_tokens / original_tokens) * 100.0

        logger.info(
            "Rewriter: accepted | %d → %d tokens (%.1f%% reduction) | similarity=%.3f",
            original_tokens, rewritten_tokens, reduction_pct, similarity
        )

        return RewriteResult(
            rewritten_query=candidate,
            candidate_query=candidate,
            original_tokens=original_tokens,
            rewritten_tokens=rewritten_tokens,
            reduction_pct=round(reduction_pct, 1),
            similarity_score=round(similarity, 3),
            fallback_used=False,
        )

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback(message: str, candidate: str, original_tokens: int, similarity: float) -> RewriteResult:
        """
        Return the original message unchanged as a safe fallback.
        Used whenever the rewrite fails any of the acceptance criteria.
        """
        return RewriteResult(
            rewritten_query=message,
            candidate_query=candidate,
            original_tokens=original_tokens,
            rewritten_tokens=original_tokens,
            reduction_pct=0.0,
            similarity_score=similarity,
            fallback_used=True,
        )
