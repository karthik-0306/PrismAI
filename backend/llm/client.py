"""
backend/llm/client.py

The single, authoritative interface for all LLM calls in PrismAI.

Every call takes an ordered *chain* of models (see backend/config.py). The client
walks the chain until one model produces a non-empty answer, and keeps a small
in-process health registry so that it stops wasting time on models it already
knows are broken:

  - 401 / 403 / 404 (bad key, project denied, model decommissioned)
        → model disabled for DEAD_COOLDOWN_S (10 min)
  - 429 rate limit
        → model skipped until the provider's "try again in Xs" has elapsed
  - timeouts / 5xx / empty output
        → short cooldown, next model is tried immediately

Without this, a dead model at the head of a chain added a failed round-trip to
*every* request, and a rate-limited model was hammered again on the very next
query — which is exactly how "the second message fails" happened.

Streaming contract (stream()):
    yields ("model", model)  — the stream for `model` opened (may later be reset)
    yields ("token", text)   — a visible text chunk
    yields ("reset", None)   — the model failed mid-answer; discard text so far,
                               a fallback model is starting over
    yields ("done",  model)  — finished; `model` is the one that produced the answer
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Tuple

import litellm

from backend import config

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True   # don't print provider banners to stdout
litellm.drop_params = True           # silently drop params a provider doesn't support

DEAD_COOLDOWN_S      = 600.0   # auth / permission / model-not-found
TRANSIENT_COOLDOWN_S = 20.0    # timeouts, 5xx, empty output
MAX_WAIT_FOR_RATE_S  = 12.0    # wait this long for a rate-limited model before skipping it
MAX_FORCED_WAIT_S    = 25.0    # ...or this long when no model in the chain is ready at all


class LLMError(Exception):
    """Raised when every model in a chain has failed."""

    def __init__(self, message: str, model: str = "", original: Optional[Exception] = None):
        super().__init__(message)
        self.model = model
        self.original = original


@dataclass
class CompletionResult:
    content: str
    model_used: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelHealth:
    cooldown_until: float = 0.0
    last_error: str = ""
    last_ok_at: float = 0.0
    last_latency_ms: int = -1
    failures: int = 0


_health: Dict[str, ModelHealth] = {}


def get_health(model: str) -> ModelHealth:
    return _health.setdefault(model, ModelHealth())


def health_snapshot() -> List[dict]:
    """Status of every configured model — used by /api/model-status (no LLM calls)."""
    now = time.monotonic()
    out = []
    for model in config.ALL_MODELS:
        h = get_health(model)
        if h.cooldown_until > now:
            status = "red" if h.cooldown_until - now > 60 else "yellow"
        elif h.last_ok_at:
            status = "green" if h.last_latency_ms < 5000 else "yellow"
        else:
            status = "unknown"
        out.append({
            "name": config.model_label(model),
            "model": model,
            "status": status,
            "latency_ms": h.last_latency_ms,
            "last_error": h.last_error[:160] if status in ("red", "yellow") else "",
        })
    return out


_TPM_RE = re.compile(r"Limit (\d+), Used (\d+), Requested (\d+)")


def _mark_ok(model: str, latency_ms: int) -> None:
    h = get_health(model)
    h.cooldown_until = 0.0
    h.last_ok_at = time.monotonic()
    h.last_latency_ms = latency_ms
    h.failures = 0
    h.last_error = ""


def _mark_failed(model: str, exc: Exception) -> None:
    h = get_health(model)
    h.failures += 1
    h.last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
    h.cooldown_until = time.monotonic() + _cooldown_for(exc)


# Groq formats waits as "2.5s", "750ms", "1m3.4s" or "12m0s".
_RETRY_AFTER_RE = re.compile(
    r"(?:try again|retry) in (?:(?P<h>\d+)h)?(?:(?P<m>\d+)m(?!s))?(?:(?P<s>[\d.]+)s|(?P<ms>[\d.]+)ms)?",
    re.IGNORECASE,
)


def _is_rate_limit(exc: Optional[Exception]) -> bool:
    return isinstance(exc, litellm.RateLimitError) or getattr(exc, "status_code", None) == 429


def _retry_after_seconds(text: str) -> Optional[float]:
    m = _RETRY_AFTER_RE.search(text)
    if not m or not any(m.group(g) for g in ("h", "m", "s", "ms")):
        return None
    return (int(m.group("h") or 0) * 3600 + int(m.group("m") or 0) * 60
            + float(m.group("s") or 0) + float(m.group("ms") or 0) / 1000.0)


def _short_error(exc: Exception) -> str:
    """One-line log summary that keeps the numbers that matter for rate limits."""
    text = str(exc)
    if _is_rate_limit(exc):
        kind = re.search(r"on (tokens|requests) per (minute|day) \((\w+)\)", text)
        tpm = _TPM_RE.search(text)
        wait = _retry_after_seconds(text)
        return (f"rate limit{' ' + kind.group(3) if kind else ''}"
                f"{f' (limit {tpm.group(1)}, used {tpm.group(2)}, requested {tpm.group(3)})' if tpm else ''}"
                f"{f', retry in {wait:.1f}s' if wait is not None else ''}")
    return f"{type(exc).__name__}: {text[:200]}"


def _cooldown_for(exc: Exception) -> float:
    text = str(exc)
    status = getattr(exc, "status_code", None)

    if _is_rate_limit(exc):
        secs = _retry_after_seconds(text)
        if secs is not None:
            return min(max(secs + 0.5, 1.0), 3600.0)
        return 30.0

    dead_markers = ("PERMISSION_DENIED", "denied access", "model_not_found",
                    "does not exist", "no longer available", "decommissioned",
                    "invalid api key", "API key not valid")
    if (isinstance(exc, (litellm.AuthenticationError, litellm.NotFoundError))
            or status in (401, 403, 404)
            or any(marker.lower() in text.lower() for marker in dead_markers)):
        return DEAD_COOLDOWN_S

    return TRANSIENT_COOLDOWN_S


MIN_USEFUL_MAX_TOKENS = 768


def _fit_to_budget(exc: Exception, max_tokens: int) -> Optional[int]:
    """
    Groq counts the *requested* max_tokens against the per-minute token budget, so
    a "rate limited" model often still has thousands of tokens free — just not the
    full reservation. Parse "Limit L, Used U, Requested R" and return a smaller
    max_tokens that fits what's left, or None if nothing useful fits.
    """
    m = _TPM_RE.search(str(exc))
    if not m:
        return None
    limit, used, requested = (int(g) for g in m.groups())
    prompt_tokens = max(0, requested - max_tokens)
    fitted = limit - used - prompt_tokens - 64
    if fitted >= MIN_USEFUL_MAX_TOKENS and fitted < max_tokens:
        return fitted
    return None


def _order_chain(models: List[str]) -> List[str]:
    """Healthy models first (original order preserved), cooling ones last by readiness."""
    now = time.monotonic()
    seen, chain = set(), []
    for m in models:
        if m and m not in seen:
            seen.add(m)
            chain.append(m)
    ready   = [m for m in chain if get_health(m).cooldown_until <= now]
    cooling = sorted((m for m in chain if get_health(m).cooldown_until > now),
                     key=lambda m: get_health(m).cooldown_until)
    return ready + cooling


async def _wait_if_cooling(model: str, force: bool = False) -> bool:
    """
    Decide whether to call a model that may be cooling down.

    - Ready → call it.
    - Rate-limited for a short while → wait it out (better than failing).
    - `force` (nothing in the chain is ready and this is the soonest-available
      model) → wait up to MAX_FORCED_WAIT_S, then call it anyway, so a request is
      never failed without a single real attempt. Waiting is skipped for models
      that are disabled for a long time (dead key / decommissioned).
    """
    remaining = get_health(model).cooldown_until - time.monotonic()
    if remaining <= 0:
        return True
    if remaining <= MAX_WAIT_FOR_RATE_S or (force and remaining <= MAX_FORCED_WAIT_S):
        logger.info("Models busy — waiting %.1fs for %s", remaining, model)
        await asyncio.sleep(remaining)
        return True
    return force


# ─────────────────────────────────────────────────────────────────────────────
# PARAM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _reasoning_effort_for(model: str) -> str:
    """
    Lowest reasoning_effort each provider accepts. These are all "thinking" models
    that spend max_tokens on invisible reasoning; for short structured calls that
    can starve the visible output entirely.
      - Qwen3.x on Groq: "none" disables thinking.
      - GPT-OSS on Groq: only low/medium/high are valid.
      - Gemini: "low" is the safest value across 2.x/3.x thinking configs.
    """
    if "qwen" in model:
        return "none"
    return "low"


def _temperature_for(model: str, requested: float) -> float:
    # Gemini 3+ degrades (and LiteLLM warns about loops) below temperature 1.0.
    return 1.0 if "gemini" in model else requested


def _build_kwargs(model: str, messages: list, temperature: float, max_tokens: int,
                  low_reasoning: bool, json_mode: bool) -> dict:
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": _temperature_for(model, temperature),
        "max_tokens": max_tokens,
        "timeout": config.LLM_TIMEOUT_S,
        "num_retries": 0,   # we handle fallback ourselves
    }
    if low_reasoning:
        kwargs["reasoning_effort"] = _reasoning_effort_for(model)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks, including an unterminated trailing one."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL)
    return text.strip()


class _ThinkFilter:
    """
    Incremental <think>…</think> remover for streamed text. Tags may be split
    across chunks, so a possible partial tag at the end of the buffer is held
    back until the next chunk disambiguates it.
    """
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self._buf = ""
        self._in_think = False

    def feed(self, piece: str) -> str:
        self._buf += piece
        out = []
        while self._buf:
            if self._in_think:
                idx = self._buf.find(self.CLOSE)
                if idx == -1:
                    # keep only a possible partial closing tag
                    self._buf = self._buf[-(len(self.CLOSE) - 1):]
                    break
                self._buf = self._buf[idx + len(self.CLOSE):].lstrip()
                self._in_think = False
            else:
                idx = self._buf.find(self.OPEN)
                if idx != -1:
                    out.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self.OPEN):]
                    self._in_think = True
                    continue
                hold = self._partial_suffix(self._buf, self.OPEN)
                out.append(self._buf[:len(self._buf) - hold])
                self._buf = self._buf[len(self._buf) - hold:]
                break
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._in_think else self._buf
        self._buf = ""
        return rest

    @staticmethod
    def _partial_suffix(text: str, tag: str) -> int:
        for n in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:n]):
                return n
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """Stateless wrapper around litellm with chain fallback. Safe to share."""

    async def complete(
        self,
        models: List[str],
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        low_reasoning: bool = False,
        json_mode: bool = False,
    ) -> CompletionResult:
        """Return the first non-empty completion from the chain, or raise LLMError."""
        last_error: Optional[Exception] = None
        last_model = models[0] if models else ""

        for round_no in range(2):
            if round_no == 1 and not await _wait_for_rate_limit_round(models, last_error):
                break
            chain = _order_chain(models)
            attempted = False
            for i, model in enumerate(chain):
                if not await _wait_if_cooling(model, force=not attempted and i == 0):
                    continue
                attempted = True
                last_model = model
                budget, shrinks = max_tokens, 0
                t0 = time.monotonic()
                try:
                    while True:
                        try:
                            logger.info("LLM call: %s (%d messages, max_tokens=%d)", model, len(messages), budget)
                            response = await litellm.acompletion(
                                **_build_kwargs(model, messages, temperature, budget, low_reasoning, json_mode)
                            )
                            break
                        except Exception as e:
                            fitted = _fit_to_budget(e, budget)
                            if not fitted or shrinks >= 3:
                                raise
                            shrinks += 1
                            logger.info("LLM %s: shrinking max_tokens %d → %d to fit rate budget", model, budget, fitted)
                            budget = fitted

                    content = strip_think_tags(response.choices[0].message.content or "")
                    if not content:
                        raise LLMError(f"{model} returned an empty response", model=model)

                    latency = int((time.monotonic() - t0) * 1000)
                    _mark_ok(model, latency)
                    usage = getattr(response, "usage", None)
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                    logger.info("LLM ok: %s | %dms | %d in / %d out", model, latency, prompt_tokens, completion_tokens)
                    return CompletionResult(
                        content=content,
                        model_used=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
                except Exception as e:
                    last_error = e
                    _mark_failed(model, e)
                    logger.warning("LLM failed: %s — %s", model, _short_error(e))

        raise LLMError(_friendly_error(last_error), model=last_model, original=last_error)

    async def stream(
        self,
        models: List[str],
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        low_reasoning: bool = False,
    ) -> AsyncIterator[Tuple[str, Optional[str]]]:
        """Stream from the first model in the chain that works. See module docstring."""
        last_error: Optional[Exception] = None
        last_model = models[0] if models else ""

        for round_no in range(2):
            if round_no == 1 and not await _wait_for_rate_limit_round(models, last_error):
                break
            chain = _order_chain(models)
            attempted = False
            for i, model in enumerate(chain):
                if not await _wait_if_cooling(model, force=not attempted and i == 0):
                    continue
                attempted = True
                last_model = model
                emitted = False
                budget, shrinks = max_tokens, 0
                t0 = time.monotonic()
                try:
                    while True:
                        try:
                            logger.info("LLM stream: %s (%d messages, max_tokens=%d)", model, len(messages), budget)
                            response = await litellm.acompletion(
                                stream=True,
                                **_build_kwargs(model, messages, temperature, budget, low_reasoning, False),
                            )
                            break
                        except Exception as e:
                            fitted = _fit_to_budget(e, budget)
                            if not fitted or shrinks >= 3:
                                raise
                            shrinks += 1
                            logger.info("LLM %s: shrinking max_tokens %d → %d to fit rate budget", model, budget, fitted)
                            budget = fitted

                    yield "model", model
                    think = _ThinkFilter()
                    async for chunk in response:
                        if not chunk.choices:
                            continue
                        piece = chunk.choices[0].delta.content or ""
                        if not piece:
                            continue
                        visible = think.feed(piece)
                        if visible:
                            if not emitted:
                                visible = visible.lstrip()
                                if not visible:
                                    continue
                            emitted = True
                            yield "token", visible
                    tail = think.flush()
                    if tail:
                        emitted = True
                        yield "token", tail
                    if not emitted:
                        raise LLMError(f"{model} streamed an empty response", model=model)

                    _mark_ok(model, int((time.monotonic() - t0) * 1000))
                    yield "done", model
                    return
                except (asyncio.CancelledError, GeneratorExit):
                    raise   # client went away — not the model's fault
                except Exception as e:
                    last_error = e
                    _mark_failed(model, e)
                    logger.warning("LLM stream failed: %s — %s", model, _short_error(e))
                    if emitted:
                        yield "reset", None

        raise LLMError(_friendly_error(last_error), model=last_model, original=last_error)


async def _wait_for_rate_limit_round(models: List[str], last_error: Optional[Exception]) -> bool:
    """
    After a full pass in which every model failed: if the last failure was a rate
    limit and some model frees up within MAX_FORCED_WAIT_S, wait for it and allow
    one more pass. Users would rather see a few extra seconds of "Thinking…" than an error.
    """
    if not _is_rate_limit(last_error):
        return False
    now = time.monotonic()
    soonest = min((get_health(m).cooldown_until for m in models), default=now) - now
    if soonest > MAX_FORCED_WAIT_S:
        return False
    if soonest > 0:
        logger.info("All models rate limited — waiting %.1fs before one more pass", soonest)
        await asyncio.sleep(soonest)
    return True


def _friendly_error(exc: Optional[Exception]) -> str:
    """A message safe to show end users (no org ids, no raw provider JSON)."""
    if exc is None:
        return "No AI model is currently available. Please try again in a minute."
    if isinstance(exc, litellm.RateLimitError) or getattr(exc, "status_code", None) == 429:
        return "All AI models are busy (free-tier rate limit reached). Please wait a few seconds and try again."
    if isinstance(exc, (litellm.Timeout, asyncio.TimeoutError)):
        return "The AI models took too long to respond. Please try again."
    if isinstance(exc, litellm.ContextWindowExceededError):
        return "This message is too long for the available models. Try shortening it."
    return "All AI models failed to respond. Please try again in a minute."
