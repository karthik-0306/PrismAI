"""
backend/utils/token_counter.py

Estimates token counts for strings before they are sent to an LLM.

Why estimate instead of using the API's actual count?
  - Token counting must happen BEFORE the API call (to decide whether to
    compress memory, choose a model, etc.).
  - tiktoken is fast, local, and accurate for OpenAI-compatible models.
  - Gemini uses a different tokenizer; we approximate with a character-based
    heuristic (4 chars ≈ 1 token) which is close enough for threshold decisions.

This module is a pure function module — no async, no side effects, no imports
from other PrismAI modules. It is safe to import from anywhere.
"""

import logging
from functools import lru_cache  # cache encoder objects — loading tiktoken encoders is slow

import tiktoken  # OpenAI's open-source tokenizer; accurate for GPT and Llama family

logger = logging.getLogger(__name__)

# ── Models whose tokenizer tiktoken can load directly ────────────────────────
# tiktoken supports the cl100k_base encoding used by GPT-3.5/4 and Llama 3 family.
# We use it as the best available approximation for all non-Gemini models.
_TIKTOKEN_ENCODING = "cl100k_base"

# ── Approximate chars-per-token for Gemini models ────────────────────────────
# Google uses SentencePiece internally; 4 chars ≈ 1 token is the standard heuristic.
_GEMINI_CHARS_PER_TOKEN = 4

# ── Providers that need tiktoken vs heuristic ────────────────────────────────
_TIKTOKEN_PROVIDERS = ("groq/", "openai/")  # LiteLLM prefix → use tiktoken
_GEMINI_PROVIDERS = ("gemini/",)             # LiteLLM prefix → use char heuristic


@lru_cache(maxsize=1)
def _get_encoder():
    """
    Load and cache the tiktoken encoder once.
    lru_cache(maxsize=1) means this only runs once per process lifetime.

    Returns:
        tiktoken.Encoding: the cl100k_base encoder object.
    """
    return tiktoken.get_encoding(_TIKTOKEN_ENCODING)


def count_tokens(text: str, model: str = "") -> int:
    """
    Estimate the number of tokens in text for a given LiteLLM model string.

    Strategy:
      - If the model starts with a groq/ or openai/ prefix → use tiktoken.
      - If the model starts with gemini/ → use the 4-chars-per-token heuristic.
      - If model is empty or unrecognized → default to tiktoken (safe choice).

    Args:
        text:  The string to count tokens for.
        model: LiteLLM model string e.g. "groq/llama-3.1-8b-instant"
               or "gemini/gemini-3.5-flash". Can be empty string.
    Returns:
        int: estimated token count. Always >= 1 for non-empty text.
    """
    if not text:
        return 0  # empty string has zero tokens

    # Check if this is a Gemini model — use char heuristic
    if any(model.startswith(prefix) for prefix in _GEMINI_PROVIDERS):
        estimated = max(1, len(text) // _GEMINI_CHARS_PER_TOKEN)
        logger.debug("Gemini char-heuristic: %d tokens for %d chars", estimated, len(text))
        return estimated

    # For Groq, OpenAI, and unknown models — use tiktoken
    try:
        encoder = _get_encoder()
        token_count = len(encoder.encode(text))
        logger.debug("tiktoken: %d tokens", token_count)
        return token_count
    except Exception as e:
        # If tiktoken fails for any reason, fall back to heuristic rather than crashing
        logger.warning("tiktoken failed (%s), falling back to char heuristic", e)
        return max(1, len(text) // _GEMINI_CHARS_PER_TOKEN)


def count_messages_tokens(messages: list, model: str = "") -> int:
    """
    Sum token counts across a list of message dicts (as sent to LiteLLM).
    Each item in messages is expected to have a 'content' key.

    Args:
        messages: list of dicts like [{"role": "user", "content": "..."}, ...]
        model:    LiteLLM model string for tokenizer selection.
    Returns:
        int: total estimated token count across all messages.
    """
    return sum(count_tokens(msg.get("content", ""), model) for msg in messages)
