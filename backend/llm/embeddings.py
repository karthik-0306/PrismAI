"""
backend/llm/embeddings.py

Computes cosine similarity between two text strings using a local
sentence-transformers model — no API key, no quota, no network dependency
after first run.

Model used: all-MiniLM-L6-v2
  - Size:       ~90 MB (downloaded once, cached permanently in ~/.cache/huggingface)
  - Dimensions: 384
  - Speed:      ~5–20 ms per encoding on CPU (after model is loaded)
  - Quality:    True semantic similarity — understands paraphrases and synonyms,
                not just keyword overlap.

Model loading:
  - First run: ~8 minutes to download. Already done. Never again.
  - Subsequent runs: ~1 second to load from disk cache. Then 5–20 ms per call.
  - The singleton (_model) is loaded on first call and reused for the entire
    server lifetime. No repeated I/O.

Usage:
    from backend.llm.embeddings import get_similarity
    score = await get_similarity("explain quicksort", "describe quicksort algorithm")
    # score ≈ 0.86 — true semantic similarity, not just keyword overlap
"""

import asyncio
import logging
import math
import os
import threading
from typing import List

logger = logging.getLogger(__name__)

# ── Model identifier ──────────────────────────────────────────────────────────
_MODEL_NAME = "all-MiniLM-L6-v2"

# Set PRISMAI_EMBEDDINGS=off on memory-constrained hosts (torch alone is ~300MB+);
# the rewriter then simply passes queries through unchanged.
ENABLED = os.getenv("PRISMAI_EMBEDDINGS", "on").strip().lower() not in ("off", "0", "false", "no")

# ── Model singleton ───────────────────────────────────────────────────────────
# Importing torch + loading the model takes 10–45s on a cold machine. Doing that
# lazily inside a request froze that user's answer for the whole duration, so the
# model is warmed in a background thread at startup (warm_up_in_background) and
# callers check is_ready() instead of waiting.
_model = None
_load_lock = threading.Lock()
_load_failed = False


def _get_model():
    """Return the SentenceTransformer model, loading it if needed (blocking)."""
    global _model, _load_failed
    if _model is None:
        with _load_lock:
            if _model is None:
                try:
                    logger.info("Loading sentence-transformers model: %s ...", _MODEL_NAME)
                    from sentence_transformers import SentenceTransformer
                    _model = SentenceTransformer(_MODEL_NAME)
                    logger.info("sentence-transformers model loaded successfully.")
                except Exception:
                    _load_failed = True
                    raise
    return _model


def is_ready() -> bool:
    return _model is not None


def warm_up_in_background() -> None:
    """Start loading the model in a daemon thread; never blocks the caller."""
    if not ENABLED or _model is not None or _load_failed or _load_lock.locked():
        return

    def _load():
        try:
            _get_model()
        except Exception as e:
            logger.warning("Embedding model failed to load (%s) — rewriter will pass queries through.", e)

    threading.Thread(target=_load, name="embedding-warmup", daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

async def get_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two text strings using dense semantic
    embeddings (all-MiniLM-L6-v2).

    Runs the CPU-bound model.encode() in a thread pool executor so it doesn't
    block the asyncio event loop.

    Returns a float between 0.0 (completely unrelated) and 1.0 (identical).
    Returns 1.0 as a safe fallback if the model fails for any reason
    (rewriter always has a valid query to continue with).

    Args:
        text_a: First text string (e.g., original user query).
        text_b: Second text string (e.g., rewritten/compressed query).
    Returns:
        float: cosine similarity score in [0.0, 1.0].
    """
    try:
        loop = asyncio.get_running_loop()
        # Run synchronous encode() in a thread — avoids blocking the event loop
        similarity = await loop.run_in_executor(
            None,
            _encode_and_compare,
            text_a,
            text_b,
        )
        return similarity
    except Exception as e:
        logger.warning(
            "Embedding similarity check failed (%s: %s) — defaulting to 1.0 (safe)",
            type(e).__name__, str(e)[:200]
        )
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS (synchronous — runs inside thread executor)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_and_compare(text_a: str, text_b: str) -> float:
    """
    Synchronous: encode both texts and compute cosine similarity.
    Called via run_in_executor to avoid blocking the event loop.
    """
    model = _get_model()
    embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
    vec_a = embeddings[0].tolist()
    vec_b = embeddings[1].tolist()
    return _cosine_similarity(vec_a, vec_b)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Pure Python cosine similarity. Used internally and in tests.

    cosine_similarity = dot(a, b) / (|a| * |b|)
    """
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return max(0.0, min(1.0, dot / (mag_a * mag_b)))
