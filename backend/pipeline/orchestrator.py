"""
backend/pipeline/orchestrator.py

The central coordinator for every user request.

There is ONE pipeline — `stream()` — and the non-streaming `run()` simply
consumes it. (Previously the two were separate copies that had drifted apart:
different history handling, judging only in one, and compound queries in the
stream path bounced to /chat, which re-saved the user message and re-ran the
rewriter and router.)

Pipeline:
  1. Resolve the chat (create it, re-create a stale id, never write into a chat
     that belongs to another session). Optionally drop the last turn (regenerate).
  2. Load history BEFORE saving the new user message, then save it.
  3. Rewriter (optional) compresses the query.
  4. Manual mode → stream the chosen model.
     Auto mode   → router (with conversation context) produces 1..3 tasks:
        single   → stream the specialist (dsa / web_search / evaluate / direct LLM)
        compound → answer tasks in parallel, judge each, retry severity-3 once,
                   then stream an aggregation of the sub-answers.
  5. Save the assistant message, schedule memory summarization, emit metadata.

Events yielded (all dicts with a "type"):
  start     {chat_id, user_message_id, is_new_chat}
  status    {stage, label}            progress hints ("Searching the web…")
  route     {categories}
  sources   {sources: [{title, url}]}
  token     {content}
  reset     {}                        discard streamed text; a fallback model restarts
  metadata  {chat_id, message_id, model_used, route_category, categories_used,
             models_used, original_tokens, rewritten_tokens, reduction_pct, title, sources}
  error     {detail}
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Tuple

from backend import config
from backend.database import queries
from backend.llm.client import LLMClient, LLMError
from backend.pipeline.memory import MemoryInjector
from backend.pipeline.rewriter import QueryRewriter, RewriteResult
from backend.pipeline.router import SmartRouter
from backend.subagents.dsa_agent import DSASubagent
from backend.subagents.evaluator_agent import EvaluatorSubagent
from backend.subagents.web_search_agent import WebSearchSubagent
from backend.utils.session import generate_id
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

_llm          = LLMClient()
_memory       = MemoryInjector()
_router       = SmartRouter()
_rewriter     = QueryRewriter()
_dsa_agent    = DSASubagent()
_evaluator    = EvaluatorSubagent()
_web_searcher = WebSearchSubagent()

# Strong references to fire-and-forget tasks so they aren't garbage collected mid-run.
_background: set = set()

STATUS_LABELS = {
    "rewriting":  "Optimizing your query…",
    "routing":    "Choosing the best specialist…",
    "searching":  "Searching the web…",
    "thinking":   "Thinking…",
    "evaluating": "Running the judge panel…",
    "working":    "Working on each part…",
    "combining":  "Combining the answers…",
}


def _spawn(coro) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


def _status(stage: str, label: Optional[str] = None) -> dict:
    return {"type": "status", "stage": stage, "label": label or STATUS_LABELS.get(stage, "")}


class PipelineError(Exception):
    """Raised by run() when the stream ended with an error event."""


@dataclass
class OrchestratorResult:
    response:         str
    model_used:       str
    chat_id:          str
    message_id:       str
    route_category:   str
    categories_used:  List[str]
    models_used:      List[str]
    original_tokens:  int = 0
    rewritten_tokens: int = 0
    reduction_pct:    float = 0.0
    title:            Optional[str] = None
    sources:          List[dict] = field(default_factory=list)


@dataclass
class _Answer:
    """Accumulates what the current request produced."""
    text: str = ""
    models: List[str] = field(default_factory=list)
    streaming_model: Optional[str] = None   # model currently streaming (for stopped answers)
    sources: List[dict] = field(default_factory=list)


class Orchestrator:
    """Stateless — one instance serves all concurrent requests."""

    async def run(self, **kwargs) -> OrchestratorResult:
        """Non-streaming wrapper around stream()."""
        text, meta = "", None
        async for event in self.stream(**kwargs):
            kind = event["type"]
            if kind == "token":
                text += event["content"]
            elif kind == "reset":
                text = ""
            elif kind == "error":
                raise PipelineError(event["detail"])
            elif kind == "metadata":
                meta = event
        if meta is None:
            raise PipelineError("The pipeline finished without a response.")
        return OrchestratorResult(
            response=text,
            model_used=meta["model_used"],
            chat_id=meta["chat_id"],
            message_id=meta["message_id"],
            route_category=meta["route_category"],
            categories_used=meta["categories_used"],
            models_used=meta["models_used"],
            original_tokens=meta["original_tokens"],
            rewritten_tokens=meta["rewritten_tokens"],
            reduction_pct=meta["reduction_pct"],
            title=meta.get("title"),
            sources=meta.get("sources", []),
        )

    async def stream(
        self,
        session_id: str,
        chat_id: Optional[str],
        message: str,
        model_preference: str = "auto",
        rewriter_enabled: bool = True,
        regenerate: bool = False,
    ) -> AsyncIterator[dict]:
        message = message.strip()

        # ── 1. Resolve chat ─────────────────────────────────────────────────
        chat_id, is_new = await self._resolve_chat(session_id, chat_id, message)
        title_task = asyncio.get_running_loop().create_task(self._generate_title(chat_id, message)) if is_new else None

        if regenerate and not is_new:
            await self._drop_last_turn(chat_id)

        # ── 2. History first, then persist the user message ─────────────────
        history = await _memory.load_history(chat_id)
        user_message_id = generate_id()
        await queries.save_message(
            message_id=user_message_id, chat_id=chat_id, role="user",
            content=message, token_count=count_tokens(message),
        )
        yield {"type": "start", "chat_id": chat_id, "user_message_id": user_message_id, "is_new_chat": is_new}

        answer = _Answer()
        route_category, categories = "manual", ["manual"]
        rewrite: Optional[RewriteResult] = None
        saved = False

        try:
            # ── 3. Rewriter ─────────────────────────────────────────────────
            query = message
            if rewriter_enabled:
                if count_tokens(message) >= 40:
                    yield _status("rewriting")
                rewrite = await _rewriter.rewrite(message)
                query = rewrite.rewritten_query

            # ── 4a. Manual mode ─────────────────────────────────────────────
            if model_preference and model_preference != "auto":
                chain = [model_preference] + [m for m in config.ALL_MODELS if m != model_preference]
                yield _status("thinking")
                async for event in self._stream_llm(chain, MemoryInjector.build_prompt(history, query), answer):
                    yield event

            # ── 4b. Auto mode ───────────────────────────────────────────────
            else:
                yield _status("routing")
                tasks = await _router.classify_and_split(query, MemoryInjector.context_snippet(history))
                categories = [t["category"] for t in tasks]
                route_category = ",".join(categories)
                yield {"type": "route", "categories": categories}

                if len(tasks) == 1:
                    async for event in self._stream_single(tasks[0], query, history, answer):
                        yield event
                else:
                    async for event in self._stream_compound(tasks, query, history, answer):
                        yield event

            # ── 5. Persist + metadata ───────────────────────────────────────
            assistant_id = await self._save_answer(chat_id, answer, route_category, rewrite, message)
            saved = True
            _spawn(_memory.maybe_summarize(chat_id))

            title = None
            if title_task:
                try:
                    title = await asyncio.wait_for(asyncio.shield(title_task), timeout=4)
                except Exception:
                    title = None

            yield {
                "type":             "metadata",
                "chat_id":          chat_id,
                "message_id":       assistant_id,
                "user_message_id":  user_message_id,
                "model_used":       _primary_model(answer.models),
                "route_category":   route_category,
                "categories_used":  categories,
                "models_used":      answer.models,
                **_token_fields(rewrite, message),
                "title":            title,
                "sources":          answer.sources,
            }

        except LLMError as e:
            logger.error("Pipeline failed for chat %s: %s", chat_id, e)
            yield {"type": "error", "detail": str(e), "chat_id": chat_id}
        except (asyncio.CancelledError, GeneratorExit):
            # Client hit "stop" or disconnected: keep whatever was generated so far.
            if answer.text.strip() and not saved:
                if not answer.models and answer.streaming_model:
                    answer.models.append(answer.streaming_model)
                logger.info("Stream cancelled for chat %s — saving partial answer", chat_id)
                _spawn(self._save_answer(chat_id, answer, route_category, rewrite, message))
            raise
        except Exception as e:
            logger.exception("Unexpected pipeline error for chat %s", chat_id)
            yield {"type": "error", "detail": "Something went wrong while generating the answer. Please try again.",
                   "chat_id": chat_id}

    # ─────────────────────────────────────────────────────────────────────────
    # ROUTES
    # ─────────────────────────────────────────────────────────────────────────

    async def _stream_single(self, task: dict, query: str, history: List[dict], answer: _Answer):
        category, sub_query = task["category"], task["sub_query"]

        if category == "evaluate":
            yield _status("evaluating")
            report, models = await _evaluator.evaluate(query, history)
            answer.models = models or ["evaluator"]
            for i in range(0, len(report), 24):
                answer.text += report[i:i + 24]
                yield {"type": "token", "content": report[i:i + 24]}
            return

        if category == "web_search":
            yield _status("searching")
            messages, sources = await _web_searcher.prepare(sub_query, history)
            answer.sources = sources
            if sources:
                yield {"type": "sources", "sources": sources}
            yield _status("thinking", "Reading the results…")
            async for event in self._stream_llm(_web_searcher.chain, messages, answer):
                yield event
            return

        yield _status("thinking")
        if category == "dsa":
            messages, chain = _dsa_agent.build_messages(query, history), _dsa_agent.chain
        else:
            messages = MemoryInjector.build_prompt(history, query)
            chain = config.ROUTE_CHAINS.get(category, config.ROUTE_CHAINS["general"])
        # Quick questions don't benefit from deep thinking — it only delays the first token.
        async for event in self._stream_llm(chain, messages, answer, low_reasoning=category == "fast"):
            yield event

    async def _stream_compound(self, tasks: List[dict], query: str, history: List[dict], answer: _Answer):
        yield _status("working", f"Working on {len(tasks)} parts in parallel…")
        if any(t["category"] == "web_search" for t in tasks):
            yield _status("searching")

        results = await asyncio.gather(*[self._answer_task(t, history) for t in tasks], return_exceptions=True)

        parts: List[dict] = []
        for task, res in zip(tasks, results):
            if isinstance(res, Exception):
                logger.warning("Sub-task %s failed: %s", task["category"], res)
                continue
            text, model, sources = res
            parts.append({**task, "response": text})
            answer.models.append(model)
            answer.sources.extend(s for s in sources if s not in answer.sources)

        if not parts:
            raise LLMError("All AI models failed to respond. Please try again in a minute.")
        if answer.sources:
            yield {"type": "sources", "sources": answer.sources}

        yield _status("combining")
        blocks = "\n\n".join(
            f"--- Part {i} ({p['category']}) ---\nTask: {p['sub_query']}\nAnswer:\n{p['response']}"
            for i, p in enumerate(parts, 1)
        )
        failed = len(tasks) - len(parts)
        prompt = (
            f"User's message:\n{query}\n\nSpecialist answers to its parts:\n{blocks}\n\n"
            "Write ONE final answer to the user's message that integrates these parts naturally, "
            "with a clear heading per distinct topic. Keep all code, numbers and citations [n] intact. "
            "Point out any contradiction between parts. Do not mention 'parts' or 'specialists'."
            + (f"\nNote: {failed} part(s) could not be answered — tell the user briefly." if failed else "")
        )
        aggregator = _Answer()
        try:
            async for event in self._stream_llm(
                config.ROUTE_CHAINS["general"],
                [{"role": "system", "content": "You merge expert answers into one clear, well-formatted reply."},
                 {"role": "user", "content": prompt}],
                aggregator,
                max_tokens=config.ANSWER_MAX_TOKENS,
            ):
                yield event
            answer.text = aggregator.text
        except LLMError:
            logger.warning("Aggregation failed — falling back to concatenated parts")
            if aggregator.text:
                yield {"type": "reset"}
            joined = "\n\n---\n\n".join(f"### {p['category'].replace('_', ' ').title()}\n\n{p['response']}" for p in parts)
            answer.text = joined
            yield {"type": "token", "content": joined}

    async def _answer_task(self, task: dict, history: List[dict]) -> Tuple[str, str, List[dict]]:
        """Non-streaming answer for one compound sub-task, with judge + one retry."""
        text, model, sources = await self._dispatch(task["category"], task["sub_query"], history)
        if task["category"] in ("evaluate", "web_search"):
            return text, model, sources

        judgment = await self._judge(task["sub_query"], text)
        if judgment["severity"] < 3:
            return text, model, sources

        logger.warning("Severity 3 for %s (%s) — retrying once", task["category"], judgment["reason"])
        retry_query = (f"{task['sub_query']}\n\n(A previous attempt was rejected because: "
                       f"{judgment['reason']}. Address that fully.)")
        try:
            return await self._dispatch(task["category"], retry_query, history)
        except LLMError:
            return text, model, sources

    async def _dispatch(self, category: str, sub_query: str, history: List[dict]) -> Tuple[str, str, List[dict]]:
        if category == "evaluate":
            report, models = await _evaluator.evaluate(sub_query, history)
            return report, (models[0] if models else "evaluator"), []
        if category == "web_search":
            messages, sources = await _web_searcher.prepare(sub_query, history)
            result = await _llm.complete(_web_searcher.chain, messages, max_tokens=config.ANSWER_MAX_TOKENS)
            return result.content, result.model_used, sources
        if category == "dsa":
            messages, chain = _dsa_agent.build_messages(sub_query, history), _dsa_agent.chain
        else:
            messages = MemoryInjector.build_prompt(history, sub_query)
            chain = config.ROUTE_CHAINS.get(category, config.ROUTE_CHAINS["general"])
        result = await _llm.complete(chain, messages, max_tokens=config.ANSWER_MAX_TOKENS)
        return result.content, result.model_used, []

    async def _judge(self, sub_query: str, response: str) -> dict:
        """Severity 0-3 from the utility chain. Never raises (failure = accept)."""
        prompt = (
            f"Task: {sub_query[:2000]}\n\nResponse:\n{response[:6000]}\n\n"
            "Rate the response: 0 = fully correct, 1 = minor gaps, 2 = partial, "
            "3 = off-topic, empty or clearly wrong.\n"
            'Respond with JSON only: {"severity": <0-3>, "reason": "<one sentence>"}'
        )
        try:
            result = await _llm.complete(config.UTILITY_CHAIN, [{"role": "user", "content": prompt}],
                                         temperature=0.0, max_tokens=150, low_reasoning=True, json_mode=True)
            raw = result.content
            parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            return {"severity": max(0, min(3, int(parsed.get("severity", 0)))),
                    "reason": str(parsed.get("reason", ""))}
        except Exception as e:
            logger.debug("Judge failed (%s) — accepting", e)
            return {"severity": 0, "reason": "judge unavailable"}

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    async def _stream_llm(self, chain: List[str], messages: List[dict], answer: _Answer,
                          max_tokens: int = config.ANSWER_MAX_TOKENS, low_reasoning: bool = False):
        """Forward LLMClient.stream() events, keeping `answer` in sync with what the user sees."""
        async for kind, value in _llm.stream(chain, messages, max_tokens=max_tokens, low_reasoning=low_reasoning):
            if kind == "model":
                answer.streaming_model = value
            elif kind == "token":
                answer.text += value
                yield {"type": "token", "content": value}
            elif kind == "reset":
                answer.text = ""
                yield {"type": "reset"}
            elif kind == "done":
                answer.models.append(value)

    async def _resolve_chat(self, session_id: str, chat_id: Optional[str], message: str) -> Tuple[str, bool]:
        if chat_id:
            chat = await queries.get_chat(chat_id)
            if chat and chat.session_id == session_id:
                return chat_id, False
            if chat:
                # Someone else's chat id — never write into it.
                logger.warning("chat %s does not belong to session %s — starting a new chat", chat_id, session_id)
                chat_id = None
        # New chat, or a stale id from the browser after the server's DB was reset:
        # re-create it under the same id so the frontend's state stays valid.
        chat_id = chat_id or generate_id()
        await queries.save_chat(chat_id, session_id, _fallback_title(message))
        logger.info("Created chat %s for session %s", chat_id, session_id)
        return chat_id, True

    async def _drop_last_turn(self, chat_id: str) -> None:
        """Regenerate: remove the trailing assistant reply and the question it answered."""
        messages = await queries.get_chat_messages(chat_id)
        to_delete = []
        if messages and messages[-1].role == "assistant":
            to_delete.append(messages.pop().message_id)
        if messages and messages[-1].role == "user":
            to_delete.append(messages.pop().message_id)
        await queries.delete_messages(to_delete)

    async def _save_answer(self, chat_id: str, answer: _Answer, route_category: str,
                           rewrite: Optional[RewriteResult], original: str) -> str:
        message_id = generate_id()
        await queries.save_message(
            message_id=message_id,
            chat_id=chat_id,
            role="assistant",
            content=answer.text,
            token_count=count_tokens(answer.text),
            model_used=_primary_model(answer.models),
            route_category=route_category,
            models_used=answer.models,
            **_token_fields(rewrite, original),
        )
        logger.info("Saved answer %s | route=%s | models=%s", message_id, route_category, answer.models)
        return message_id

    async def _generate_title(self, chat_id: str, message: str) -> Optional[str]:
        """Short LLM-written sidebar title for a new chat; keeps the fallback on failure."""
        try:
            result = await _llm.complete(
                config.UTILITY_CHAIN,
                [{"role": "user", "content":
                  "Write a short title (2-6 words, no quotes, no trailing punctuation) for a chat "
                  f"that starts with this message:\n\n{message[:1000]}"}],
                temperature=0.3, max_tokens=24, low_reasoning=True,
            )
            title = result.content.strip().strip('"\'').splitlines()[0].strip().rstrip(".")[:60]
            if title:
                await queries.rename_chat(chat_id, title)
                return title
        except Exception as e:
            logger.debug("Title generation failed: %s", e)
        return None


def _primary_model(models: List[str]) -> str:
    if len(models) == 1:
        return models[0]
    return "aggregated" if models else "unknown"


def _fallback_title(message: str) -> str:
    words = message.split()
    title = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
    return title[:80] or "New Chat"


def _token_fields(rewrite: Optional[RewriteResult], original: str) -> dict:
    if rewrite:
        return {"original_tokens": rewrite.original_tokens,
                "rewritten_tokens": rewrite.rewritten_tokens,
                "reduction_pct": rewrite.reduction_pct}
    n = count_tokens(original)
    return {"original_tokens": n, "rewritten_tokens": n, "reduction_pct": 0.0}
