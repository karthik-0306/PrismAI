"""
backend/llm/client.py

The single, authoritative interface for all LLM calls in PrismAI.

Why centralise here?
  - Every LLM call in the project goes through LLMClient.async_complete().
  - Fallback chains are configured in one place — change a model string here
    and every pipeline stage picks it up automatically.
  - Token usage is logged here, so we have a single place to add metering later.
  - Typed exceptions are raised so callers can distinguish rate limits from
    model-not-found errors without parsing LiteLLM error strings themselves.

Architecture:
  - LLMClient is a stateless class (no instance state) — instantiate it once
    in the orchestrator, or call it as a module-level singleton.
  - async_complete() tries the primary model, then fallback_1, then fallback_2.
  - If all fail, it raises LLMError with the last exception details.

ROUTE_MAP: hardcoded here as the single source of truth for model assignments.
  If a model string needs to change, this is the only place to edit it.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import litellm                # unified interface to all LLM providers
from dotenv import load_dotenv  # load GEMINI_API_KEY, GROQ_API_KEY from .env

load_dotenv()  # load .env before any litellm calls — must happen before module use

logger = logging.getLogger(__name__)

# ── Suppress litellm's verbose stdout output in production ────────────────────
litellm.suppress_debug_info = True  # don't print to stdout on every call


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE MAP — Single source of truth for model assignments
# ─────────────────────────────────────────────────────────────────────────────
# All model strings are live-tested LiteLLM prefixed identifiers.
# Change a model here and every pipeline stage picks it up automatically.
# fallback_2 is always Gemini Flash — fastest, most permissive quota model.

ROUTE_MAP: dict[str, dict[str, str]] = {
    "dsa": {
        "primary":    "groq/openai/gpt-oss-120b",        # best reasoning on Groq
        "fallback_1": "groq/qwen/qwen3-32b",             # strong coding/logic model
        "fallback_2": "gemini/gemini-3.5-flash",         # reliable last resort
    },
    "coding": {
        "primary":    "groq/qwen/qwen3-32b",             # Qwen excels at code generation
        "fallback_1": "groq/openai/gpt-oss-120b",        # strong alternative
        "fallback_2": "gemini/gemini-3.5-flash",
    },
    "reasoning": {
        "primary":    "groq/openai/gpt-oss-120b",        # strongest free-tier reasoning
        "fallback_1": "groq/llama-3.3-70b-versatile",   # capable general fallback
        "fallback_2": "gemini/gemini-3.5-flash",
    },
    "math": {
        "primary":    "groq/openai/gpt-oss-120b",        # numerical reasoning strength
        "fallback_1": "groq/qwen/qwen3-32b",             # Qwen also strong at math
        "fallback_2": "gemini/gemini-3.5-flash",
    },
    "summarize": {
        "primary":    "gemini/gemini-3.5-flash",         # fast, cheap, good at summaries
        "fallback_1": "groq/llama-3.1-8b-instant",      # ultra-fast small model
        "fallback_2": "groq/llama-3.3-70b-versatile",
    },
    "fast": {
        "primary":    "groq/llama-3.1-8b-instant",      # lowest latency on Groq
        "fallback_1": "gemini/gemini-3.5-flash",
        "fallback_2": "groq/llama-3.3-70b-versatile",
    },
    "general": {
        "primary":    "gemini/gemini-3.5-flash",         # best all-rounder for general chat
        "fallback_1": "groq/llama-3.3-70b-versatile",
        "fallback_2": "groq/llama-3.1-8b-instant",
    },
}

# Utility model used by the rewriter and router — must be fast and cheap
UTILITY_MODEL: dict[str, str] = {
    "primary":    "groq/llama-3.1-8b-instant",
    "fallback_1": "gemini/gemini-3.5-flash",
}

# Context window sizes per model (in tokens) — used by memory injector (Phase 4)
# Values are conservative estimates; real limits are slightly higher.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "groq/openai/gpt-oss-120b":              8_000,
    "groq/qwen/qwen3-32b":                  32_000,
    "groq/llama-3.3-70b-versatile":         32_000,
    "groq/llama-3.1-8b-instant":            32_000,
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": 16_000,
    "gemini/gemini-3.5-flash":             100_000,
}


# ─────────────────────────────────────────────────────────────────────────────
# TYPED EXCEPTION
# ─────────────────────────────────────────────────────────────────────────────

class LLMError(Exception):
    """
    Raised when all models in a fallback chain have failed.
    Wraps the last underlying exception and the model string that caused it,
    so callers can log or surface a meaningful error to the frontend.
    """
    def __init__(self, message: str, model: str, original: Exception):
        super().__init__(message)
        self.model = model        # which model triggered the final failure
        self.original = original  # the raw exception from litellm


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETION RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompletionResult:
    """
    Typed return value from async_complete().
    Bundles the response text with metadata the orchestrator needs for logging.
    """
    content: str            # the assistant's reply text (think tags stripped)
    model_used: str         # which model actually produced this response
    prompt_tokens: int      # tokens in the input messages
    completion_tokens: int  # tokens in the output
    total_tokens: int       # prompt + completion


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Stateless wrapper around litellm.acompletion with automatic fallback chains.

    Usage:
        client = LLMClient()
        result = await client.async_complete(
            model="groq/qwen/qwen3-32b",
            messages=[{"role": "user", "content": "hello"}],
            fallback_models=["groq/openai/gpt-oss-120b", "gemini/gemini-3.5-flash"]
        )
    """

    async def async_complete(
        self,
        model: str,
        messages: list,
        fallback_models: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> CompletionResult:
        """
        Call the LLM with automatic fallback on failure.
        Tries `model` first. If it raises any exception, tries each model in
        `fallback_models` in order. Raises LLMError if all models fail.

        Args:
            model:           Primary LiteLLM model string to try first.
            messages:        List of {role, content} dicts — the full prompt.
            fallback_models: Ordered list of fallback model strings.
                             Defaults to empty list (no fallback).
            temperature:     Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens:      Maximum tokens in the response.
        Returns:
            CompletionResult: typed object with content and token usage.
        Raises:
            LLMError: if every model in the chain fails.
        """
        if fallback_models is None:
            fallback_models = []

        # Build the full ordered chain: primary first, then fallbacks
        all_models = [model] + fallback_models

        last_error: Optional[Exception] = None
        last_model: str = model

        for attempt_model in all_models:
            try:
                logger.info("Calling LLM: %s (%d messages)", attempt_model, len(messages))

                response = await litellm.acompletion(
                    model=attempt_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Extract the text content from the response
                raw_content = response.choices[0].message.content or ""

                # Strip <think>...</think> blocks that Qwen models emit
                # These chain-of-thought blocks should not be shown to the user
                clean_content = self._strip_think_tags(raw_content)

                # Extract token usage — default to 0 if provider doesn't report it
                usage = response.usage or {}
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

                logger.info(
                    "LLM success: %s | tokens: %d in, %d out",
                    attempt_model, prompt_tokens, completion_tokens
                )

                return CompletionResult(
                    content=clean_content,
                    model_used=attempt_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            except Exception as e:
                # Log the failure and try the next model in the chain
                last_error = e
                last_model = attempt_model
                logger.warning(
                    "LLM call failed for %s: %s: %s — trying next fallback",
                    attempt_model, type(e).__name__, str(e)[:200]
                )

        # All models exhausted — raise a typed error with context
        raise LLMError(
            f"All models failed. Last model: {last_model}. Last error: {last_error}",
            model=last_model,
            original=last_error,
        )

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """
        Remove <think>...</think> chain-of-thought blocks from model output.
        Qwen3-32b and GPT-OSS-120b emit these reasoning traces before the answer.
        The user should see only the final answer, not the thinking process.

        Handles:
          - Single <think> block at the start (most common case)
          - Multiple <think> blocks anywhere in the response
          - Malformed (unclosed) think tags — leaves content unchanged

        Args:
            text: raw LLM response text, possibly containing <think> blocks.
        Returns:
            str: cleaned text with all <think>...</think> sections removed, stripped.
        """
        import re
        # re.DOTALL makes . match newlines inside the thinking block
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()  # remove leading/trailing whitespace after stripping


def get_fallbacks_for_route(route_category: str) -> tuple[str, list]:
    """
    Look up the primary model and fallback list for a given route category.
    Convenience function used by the orchestrator so it doesn't import ROUTE_MAP directly.

    Args:
        route_category: one of 'dsa', 'coding', 'reasoning', 'math',
                        'summarize', 'fast', 'general'.
                        Defaults to 'general' if unknown.
    Returns:
        tuple: (primary_model_string, [fallback_1, fallback_2])
    """
    # Default to general if the router returns an unrecognized category
    route = ROUTE_MAP.get(route_category, ROUTE_MAP["general"])
    primary = route["primary"]
    fallbacks = [route["fallback_1"], route["fallback_2"]]
    return primary, fallbacks


def get_utility_model() -> tuple[str, list]:
    """
    Return the utility model config (used by rewriter and router classifiers).
    These tasks need speed, not capability — llama-3.1-8b-instant is ideal.

    Returns:
        tuple: (primary_model_string, [fallback_1])
    """
    return UTILITY_MODEL["primary"], [UTILITY_MODEL["fallback_1"]]
