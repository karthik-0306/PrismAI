"""
backend/llm/embeddings.py

Semantic similarity between two texts — used by the query rewriter to check
that a compressed query still means the same thing as the original.

Backends (env PRISMAI_EMBEDDINGS):
  gemini (default when GEMINI_API_KEY is set)
      Gemini embedding API (gemini-embedding-001). ~1s, no local RAM/CPU.
  local
      sentence-transformers all-MiniLM-L6-v2 (pip install sentence-transformers).
      Fully offline, but torch needs ~300MB+ RAM and 10–45s to load — it starved
      the whole server on Render's free tier (512MB, fractional CPU), which is
      why it is no longer the default or in requirements.txt.
  off
      No similarity check — the rewriter then passes queries through unchanged.

The two backends score on different scales, so each has its own acceptance
threshold (similarity_threshold()). The Gemini threshold was calibrated on
real rewrite pairs: faithful compressions scored 0.90–0.96, drifted or
"answered instead of compressed" outputs ~0.83. Over-compressed rewrites that
drop a constraint can still score ~0.90, so the rewriter also enforces a
minimum kept-token ratio (see rewriter.MIN_KEPT_RATIO).

Usage:
    from backend.llm import embeddings
    score = await embeddings.get_similarity("explain quicksort", "describe quicksort")
    # float in [0, 1], or None if the backend is unavailable
"""

import asyncio
import logging
import math
import os
import threading
from typing import List, Optional

import httpx

from backend import config  # noqa: F401 — loads .env before the backend is chosen

logger = logging.getLogger(__name__)

_GEMINI_EMBED_MODEL = os.getenv("PRISMAI_EMBED_MODEL", "models/gemini-embedding-001")
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/{model}:batchEmbedContents"
_LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"

_THRESHOLDS = {"gemini": 0.88, "local": 0.80}


def _resolve_backend() -> str:
    choice = os.getenv("PRISMAI_EMBEDDINGS", "").strip().lower()
    if choice in ("off", "0", "false", "no", "none"):
        return "off"
    if choice == "local":
        return "local"
    if choice in ("", "on", "gemini", "auto"):
        return "gemini" if os.getenv("GEMINI_API_KEY", "").strip() else "off"
    logger.warning("Unknown PRISMAI_EMBEDDINGS=%r — embeddings disabled", choice)
    return "off"


BACKEND = _resolve_backend()


def similarity_threshold() -> float:
    return _THRESHOLDS.get(BACKEND, 1.0)


def is_ready() -> bool:
    if BACKEND == "gemini":
        return True
    if BACKEND == "local":
        return _local_model is not None
    return False


async def get_similarity(text_a: str, text_b: str) -> Optional[float]:
    """Cosine similarity in [0, 1], or None if it couldn't be computed (never raises)."""
    try:
        if BACKEND == "gemini":
            vec_a, vec_b = await _gemini_embed([text_a, text_b])
        elif BACKEND == "local" and _local_model is not None:
            loop = asyncio.get_running_loop()
            vec_a, vec_b = await loop.run_in_executor(None, _local_embed, [text_a, text_b])
        else:
            return None
        return _cosine_similarity(vec_a, vec_b)
    except Exception as e:
        logger.warning("Similarity check failed (%s: %s)", type(e).__name__, str(e)[:200])
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI BACKEND
# ─────────────────────────────────────────────────────────────────────────────

async def _gemini_embed(texts: List[str]) -> List[List[float]]:
    payload = {"requests": [
        {"model": _GEMINI_EMBED_MODEL, "content": {"parts": [{"text": t[:8000]}]},
         "taskType": "SEMANTIC_SIMILARITY"}
        for t in texts
    ]}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            _GEMINI_URL.format(model=_GEMINI_EMBED_MODEL),
            params={"key": os.getenv("GEMINI_API_KEY", "")},
            json=payload,
        )
        resp.raise_for_status()
        return [e["values"] for e in resp.json()["embeddings"]]


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL BACKEND (optional)
# ─────────────────────────────────────────────────────────────────────────────

_local_model = None
_load_lock = threading.Lock()


def _local_embed(texts: List[str]) -> List[List[float]]:
    return [v.tolist() for v in _local_model.encode(texts, convert_to_numpy=True)]


def warm_up_in_background() -> None:
    """Load the local model in a daemon thread (local backend only); never blocks."""
    if BACKEND != "local" or _local_model is not None or _load_lock.locked():
        return

    def _load():
        global _local_model
        with _load_lock:
            try:
                logger.info("Loading sentence-transformers model: %s ...", _LOCAL_MODEL_NAME)
                from sentence_transformers import SentenceTransformer
                _local_model = SentenceTransformer(_LOCAL_MODEL_NAME)
                logger.info("sentence-transformers model loaded.")
            except Exception as e:
                logger.warning("Local embedding model failed to load (%s) — rewriter will pass queries through.", e)

    threading.Thread(target=_load, name="embedding-warmup", daemon=True).start()


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))
