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
      PRISMAI_MODEL_GEMINI=gemini/gemini-3.5-flash

Rate-limit design:
  Groq's free tier gives each model its own ~8000 tokens/minute budget, and the
  `max_tokens` you *request* counts against it. Spreading work across three Groq
  models (plus Gemini) triples the usable budget, so the chains below deliberately
  put utility work (routing, rewriting, titles) on a different model than answers.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

SMART  = _env("PRISMAI_MODEL_SMART",  "groq/openai/gpt-oss-120b")   # deepest reasoning
CODER  = _env("PRISMAI_MODEL_CODER",  "groq/qwen/qwen3.8-27b")      # code, fast, can disable thinking
FAST   = _env("PRISMAI_MODEL_FAST",   "groq/openai/gpt-oss-20b")    # lowest latency
GEMINI = _env("PRISMAI_MODEL_GEMINI", "gemini/gemini-3.5-flash")    # different provider = outage insurance

# Human-friendly names for the UI (model picker, badges, health dots).
MODEL_LABELS = {
    SMART:  "GPT-OSS 120B",
    CODER:  "Qwen 3.8 27B",
    FAST:   "GPT-OSS 20B",
    GEMINI: "Gemini Flash",
}

# Every model the app may call, best-first. Manual mode falls back through this.
ALL_MODELS = [SMART, CODER, GEMINI, FAST]

# Answer chains per route category: first entry is primary, the rest are fallbacks.
ROUTE_CHAINS = {
    "dsa":        [SMART, CODER, GEMINI, FAST],
    "coding":     [CODER, SMART, GEMINI, FAST],
    "reasoning":  [SMART, CODER, GEMINI, FAST],
    "math":       [SMART, CODER, GEMINI, FAST],
    "summarize":  [CODER, GEMINI, FAST, SMART],
    "fast":       [FAST, CODER, GEMINI, SMART],
    "general":    [CODER, SMART, GEMINI, FAST],
    "web_search": [CODER, SMART, GEMINI, FAST],   # synthesis of search results
}

# Internal utility work (router, rewriter, judge, aggregator, titles, summaries).
# Qwen first: it answers in ~0.2s with thinking disabled, and keeps these calls
# off the budgets of the models that write the user-visible answers.
UTILITY_CHAIN = [CODER, FAST, GEMINI, SMART]

# The evaluator is presented as a "panel of independent judges" — use genuinely
# different models so the panel isn't one model sampled three times.
JUDGE_PANEL = [SMART, CODER, FAST]


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION LIMITS
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_MAX_TOKENS = int(_env("PRISMAI_ANSWER_MAX_TOKENS", "4096"))
LLM_TIMEOUT_S     = float(_env("PRISMAI_LLM_TIMEOUT_S", "60"))

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


def model_label(model: str) -> str:
    """Short display name for a LiteLLM model string."""
    return MODEL_LABELS.get(model, model.split("/")[-1])
