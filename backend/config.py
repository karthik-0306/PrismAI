"""
backend/config.py

Single source of truth for every model string and tunable in PrismAI.

Why this file exists:
  Providers rename and decommission models constantly (qwen3-32b → qwen3.6-27b →
  qwen3.8-27b in a few months). Before this module, model strings were copied into
  seven files and drifted apart — some chains pointed at models that no longer
  existed, so every fallback silently failed. Now every module imports from here,
  and each model can be overridden with an env var without a code change:

      PRISMAI_MODEL_SMART=groq/openai/gpt-oss-120b
      PRISMAI_MODEL_CODER=groq/qwen/qwen3.8-27b
      PRISMAI_MODEL_FAST=groq/openai/gpt-oss-20b
      PRISMAI_MODEL_GEMINI=gemini/gemini-3.5-flash-lite
      PRISMAI_MODEL_NVIDIA=nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b

Rate-limit design (free tiers, verified Nov 2026):
  - Groq gives each model its own 8000 tokens/minute budget, and the `max_tokens`
    you *request* counts against it.
  - Qwen 3.8 on Groq additionally has only 1000 OUTPUT tokens/minute — one long
    answer exhausts it. So Qwen does short utility work (routing, titles, judging)
    and is the last resort for answers.
  - Gemini Flash-Lite: ~500 requests/day and a very large tokens/minute budget, so
    it absorbs long prompts (search results, texts to summarize) and long code
    answers. (Gemini Flash allows only 20 requests/day on the free tier.)
  - NVIDIA NIM (optional, enabled by NVIDIA_API_KEY): ~40 requests/minute per
    account and no per-minute token budget, but slow at writing long answers on
    the free tier (30-70s for a big code answer). Placed second in the chains it
    answered 35 of 50 test messages and the average reply went from 5s to 23s, so
    it is a late safety net instead: it answers when Groq and Gemini are exhausted,
    rather than the chat failing. Nemotron 3 Ultra was the only NIM model that was
    both reasonably fast and accurate; Kimi, GLM, Gemma, Mistral Large and Step
    took 60s+ or timed out.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# LiteLLM otherwise downloads its model price map from GitHub at import time,
# which adds seconds (or a timeout) to every cold start. The bundled copy is fine.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


# LiteLLM reads NVIDIA_NIM_API_KEY; accept the name NVIDIA's own docs use too.
if os.getenv("NVIDIA_API_KEY", "").strip() and not os.getenv("NVIDIA_NIM_API_KEY", "").strip():
    os.environ["NVIDIA_NIM_API_KEY"] = os.environ["NVIDIA_API_KEY"].strip()


def _chain(*models):
    """Drop models that aren't configured (None), keeping order."""
    return [m for m in models if m]


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

SMART  = _env("PRISMAI_MODEL_SMART",  "groq/openai/gpt-oss-120b")   # deepest reasoning
CODER  = _env("PRISMAI_MODEL_CODER",  "groq/qwen/qwen3.8-27b")      # code, fast, can disable thinking
FAST   = _env("PRISMAI_MODEL_FAST",   "groq/openai/gpt-oss-20b")    # lowest latency
GEMINI = _env("PRISMAI_MODEL_GEMINI", "gemini/gemini-3.5-flash-lite")  # different provider = outage insurance
# Optional: only used when an NVIDIA key is present, so the app runs unchanged without one.
NVIDIA = (_env("PRISMAI_MODEL_NVIDIA", "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b")
          if os.getenv("NVIDIA_NIM_API_KEY", "").strip() else None)

# Human-friendly names for the UI (model picker, badges, health dots).
MODEL_LABELS = {
    SMART:  "GPT-OSS 120B",
    CODER:  "Qwen 3.8 27B",
    FAST:   "GPT-OSS 20B",
    GEMINI: "Gemini Flash-Lite",
    NVIDIA: "Nemotron 3 Ultra",
}

# Every model the app may call, best-first. Manual mode falls back through this.
ALL_MODELS = _chain(SMART, CODER, GEMINI, FAST, NVIDIA)

# Answer chains per route category: first entry is primary, the rest are fallbacks.
# Qwen is last wherever answers can be long (see its output-token limit above).
# NVIDIA comes after the fast providers (slow but dependable — see above).
ROUTE_CHAINS = {
    "dsa":        _chain(SMART, GEMINI, FAST, NVIDIA, CODER),
    "coding":     _chain(SMART, GEMINI, FAST, NVIDIA, CODER),
    "reasoning":  _chain(SMART, GEMINI, FAST, NVIDIA, CODER),
    "math":       _chain(SMART, FAST, GEMINI, NVIDIA, CODER),
    "summarize":  _chain(GEMINI, SMART, FAST, NVIDIA, CODER),   # long input text
    "fast":       _chain(FAST, CODER, GEMINI, SMART, NVIDIA),   # short answers fit Qwen's budget
    "general":    _chain(SMART, FAST, GEMINI, NVIDIA, CODER),
    "web_search": _chain(GEMINI, SMART, FAST, NVIDIA, CODER),   # long prompt full of search results
}

# Internal utility work (router, rewriter, judge, titles, summaries).
# Qwen first: it answers in ~0.2s with thinking disabled, its outputs here are
# short, and it keeps these calls off the budgets of the answer models.
# NVIDIA only as a last resort here: it reasons before answering, which suits
# full answers better than these tiny-budget calls.
UTILITY_CHAIN = _chain(CODER, FAST, GEMINI, SMART, NVIDIA)

# The evaluator is presented as a "panel of independent judges" — use genuinely
# different models so the panel isn't one model sampled three times.
JUDGE_PANEL = [SMART, CODER, FAST]


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION LIMITS
# ─────────────────────────────────────────────────────────────────────────────

# Output allowance per answer. Groq counts the *requested* allowance against the
# per-minute budget, so a smaller one lets more requests through per minute.
# Measured answer lengths: median ~500 tokens, 95th percentile ~1900, with the
# long tail almost entirely code — so code-heavy routes keep the larger allowance.
ANSWER_MAX_TOKENS      = int(_env("PRISMAI_ANSWER_MAX_TOKENS", "2048"))
LONG_ANSWER_MAX_TOKENS = int(_env("PRISMAI_LONG_ANSWER_MAX_TOKENS", "4096"))
LONG_ANSWER_ROUTES     = {"coding", "dsa"}

# Timeouts. A provider that accepts a request and then stalls (Gemini sometimes
# returns 503 only after ~55s) must not freeze the chat, so a stream that sends
# nothing for too long is abandoned and the next model in the chain takes over.
#   - first chunk: all current models send one within ~1-5s (GPT-OSS streams its
#     reasoning; Flash-Lite does not think), so 20s is a generous margin
#   - between chunks: once text is flowing, a long silence means a stalled stream
LLM_TIMEOUT_S                = float(_env("PRISMAI_LLM_TIMEOUT_S", "60"))
STREAM_FIRST_CHUNK_TIMEOUT_S = float(_env("PRISMAI_STREAM_FIRST_CHUNK_TIMEOUT_S", "20"))
STREAM_STALL_TIMEOUT_S       = float(_env("PRISMAI_STREAM_STALL_TIMEOUT_S", "20"))

# Hard ceiling on a single user message (characters). Keeps one paste from
# blowing through every model's per-minute token budget at once.
MAX_MESSAGE_CHARS = int(_env("PRISMAI_MAX_MESSAGE_CHARS", "24000"))


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY
# ─────────────────────────────────────────────────────────────────────────────

# Token budget for verbatim history injected into each prompt. Most recent turns
# win; older turns drop out (and get folded into the rolling summary).
HISTORY_TOKEN_BUDGET = int(_env("PRISMAI_HISTORY_TOKEN_BUDGET", "3000"))

# When unsummarized history exceeds this many tokens, older turns are compressed
# into a summary in the background.
SUMMARIZE_TRIGGER_TOKENS = int(_env("PRISMAI_SUMMARIZE_TRIGGER_TOKENS", "4000"))

# Number of most recent messages that always stay verbatim (never summarized).
KEEP_RECENT_MESSAGES = 6


# ─────────────────────────────────────────────────────────────────────────────
# STORAGE / SERVER
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = Path(_env("PRISMAI_DB_PATH", str(_PROJECT_ROOT / "prismai.db")))

# Comma-separated list of extra allowed CORS origins (e.g. your Vercel URL).
FRONTEND_URLS = [u.strip().rstrip("/") for u in os.getenv("FRONTEND_URL", "").split(",") if u.strip()]


def answer_max_tokens(category: str) -> int:
    """Output allowance for an answer in this route category."""
    return LONG_ANSWER_MAX_TOKENS if category in LONG_ANSWER_ROUTES else ANSWER_MAX_TOKENS


def model_label(model: str) -> str:
    """Short display name for a LiteLLM model string."""
    return MODEL_LABELS.get(model, model.split("/")[-1])
