"""
backend/pipeline/orchestrator.py  (Phase 3 — Full Rewrite)

The central coordinator for every user request.

Phase 3 pipeline (5 steps):

  Step 1 — Resolve/create chat + save user message to DB

  Step 2 — Route decision:
    a) MANUAL MODE  (model_preference != "auto"):
         Skip router, skip splitting, skip judging.
         Call the chosen model directly with the full query.
         Return response immediately.

    b) AUTO MODE — single intent (router returns 1 sub-query):
         Dispatch sub-query to its ROUTE_MAP model.
         Judge the response (severity 0/1/2 = accept, 3 = retry).
         Return response directly (no aggregation step).

    c) AUTO MODE — compound intent (router returns 2+ sub-queries):
         Dispatch all sub-queries in parallel (asyncio.gather).
         Judge each response in parallel.
         Retry any severity-3 failures (corrected sub-query → same model, once).
         Aggregate all accepted responses into one final reply.

  Step 3 — Save assistant response to DB
  Step 4 — Return OrchestratorResult to HTTP router

Hard limits:
  - Max 2 calls to any expensive model per sub-task (original + 1 retry).
  - No second judgment after a retry — retry result is always accepted.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from backend.database import queries
from backend.llm.client import LLMClient
from backend.pipeline.memory import MemoryInjector
from backend.pipeline.router import SmartRouter
from backend.pipeline.rewriter import QueryRewriter, RewriteResult
from backend.subagents.dsa_agent import DSASubagent
from backend.subagents.evaluator_agent import EvaluatorSubagent
from backend.subagents.web_search_agent import WebSearchSubagent
from backend.utils.session import generate_id
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ROUTING TABLE
# Maps each category to a primary model + two fallback models.
# The LLMClient tries them in order automatically.
# ─────────────────────────────────────────────────────────────────────────────
ROUTE_MAP = {
    "dsa":       {"primary": "groq/openai/gpt-oss-120b",       "fallback_1": "groq/qwen/qwen3-32b",          "fallback_2": "gemini/gemini-3.5-flash"},
    "coding":    {"primary": "groq/qwen/qwen3-32b",            "fallback_1": "groq/openai/gpt-oss-120b",     "fallback_2": "gemini/gemini-3.5-flash"},
    "reasoning": {"primary": "groq/openai/gpt-oss-120b",       "fallback_1": "groq/llama-3.3-70b-versatile", "fallback_2": "gemini/gemini-3.5-flash"},
    "math":      {"primary": "groq/openai/gpt-oss-120b",       "fallback_1": "groq/qwen/qwen3-32b",          "fallback_2": "gemini/gemini-3.5-flash"},
    "summarize": {"primary": "gemini/gemini-3.5-flash",        "fallback_1": "groq/llama-3.1-8b-instant",    "fallback_2": "groq/llama-3.3-70b-versatile"},
    "fast":      {"primary": "groq/llama-3.1-8b-instant",      "fallback_1": "gemini/gemini-3.5-flash",      "fallback_2": "groq/llama-3.3-70b-versatile"},
    "general":   {"primary": "gemini/gemini-3.5-flash",        "fallback_1": "groq/llama-3.3-70b-versatile", "fallback_2": "groq/llama-3.1-8b-instant"},
    # web_search is handled entirely by WebSearchSubagent (no direct LLM route needed here)
}

# Priority order: cheapest/fastest first, most capable last.
# All models are listed so that if the entire Groq service is down,
# Gemini picks it up, and vice versa. Nothing is left unprotected.
UTILITY_PRIMARY   = "gemini/gemini-3.5-flash"
UTILITY_FALLBACKS = [
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.3-70b-versatile",     # stronger than 8b, still cheap on Groq
    "groq/qwen/qwen3-32b",              # strong reasoning, good JSON compliance
    "groq/openai/gpt-oss-120b",         # most capable — last resort for internal tasks
]

# ── Manual mode fallback pool — all available models sorted best-to-cheapest
# When a user manually selects a model and it fails, we try the rest in this order.
# The chosen model is excluded at runtime so it is never called twice.
# The user always sees which model ACTUALLY responded via model_used in the response.
ALL_MODELS_BY_QUALITY = [
    "groq/openai/gpt-oss-120b",
    "groq/qwen/qwen3-32b",
    "groq/llama-3.3-70b-versatile",
    "gemini/gemini-3.5-flash",
    "groq/llama-3.1-8b-instant",
]

# ── Module-level singletons — stateless, safe to share across async requests ──
_llm_client = LLMClient()
_memory     = MemoryInjector()
_router     = SmartRouter()
_rewriter   = QueryRewriter()
_dsa_agent    = DSASubagent()
_evaluator    = EvaluatorSubagent()
_web_searcher = WebSearchSubagent()


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorResult:
    """
    Everything the HTTP router (chat.py) needs to build the API response.
    """
    response:        str               # Final reply text shown to the user
    model_used:      str               # Primary model (or "aggregated" if compound)
    chat_id:         str               # UUID4 of the conversation
    message_id:      str               # UUID4 of the saved assistant message row
    route_category:  str               # e.g. "dsa" | "dsa,math" | "manual" | "general"
    categories_used: List[str]         # All categories that were dispatched
    models_used:     List[str]         # All model strings used to produce sub-responses
    original_tokens: int = 0           # Phase 2 metrics
    rewritten_tokens: int = 0          # Phase 2 metrics
    reduction_pct:   float = 0.0       # Phase 2 metrics


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Coordinates the full request-response pipeline for one user message.
    Stateless — a single instance handles all concurrent requests safely.
    """

    async def run(
        self,
        session_id:        str,
        chat_id:           Optional[str],
        message:           str,
        model_preference:  str = "auto",
        rewriter_enabled:  bool = True,
    ) -> OrchestratorResult:
        """
        Execute the Phase 3 pipeline for one user message.

        Args:
            session_id:       UUID4 string — the browser's session identifier.
            chat_id:          UUID4 string of an existing chat, or None to start a new one.
            message:          The user's raw input text.
            model_preference: "auto" uses the smart router; any other string is a manual
                              model override that bypasses routing entirely.
            rewriter_enabled: If True, the query rewriter runs FIRST regardless of
                              auto/manual mode. Phase 2 will hook rewriter.rewrite() here.
                              In Phase 3 (no rewriter built yet), this flag is accepted
                              but has no effect — the rewriter slot is reserved.
        Returns:
            OrchestratorResult with the final response and all metadata.
        """

        # ── Step 1: Resolve or create chat ─────────────────────────────────
        if chat_id is None:
            chat_id = generate_id()
            title   = self._make_title(message)
            await queries.save_chat(chat_id, session_id, title)
            logger.info("Created new chat %s for session %s", chat_id, session_id)

        # Save the user's message to DB before doing anything else
        user_message_id  = generate_id()
        user_token_count = count_tokens(message)
        await queries.save_message(
            message_id=user_message_id,
            chat_id=chat_id,
            role="user",
            content=message,
            token_count=user_token_count,
        )
        logger.debug("Saved user message %s (%d tokens)", user_message_id, user_token_count)

        # ── PHASE 2 REWRITER HOOK ──────────────────────────────────────────────
        # The rewriter compresses the query using the 70B model before
        # anything else happens.
        if rewriter_enabled:
            logger.info("Running query rewriter on user message...")
            rewrite_result = await _rewriter.rewrite(message)
            query_to_use = rewrite_result.rewritten_query
        else:
            logger.info("Rewriter disabled for this request.")
            rewrite_result = None
            query_to_use = message
        # ───────────────────────────────────────────────────────────────────

        # ── Step 2a: MANUAL MODE — user chose a specific model ───────────────────
        if model_preference != "auto":
            return await self._run_manual(
                chat_id=chat_id,
                message=query_to_use,
                model=model_preference,
                rewrite_result=rewrite_result,
            )

        # ── Step 2b/c: AUTO MODE — classify intent, dispatch, judge, aggregate
        return await self._run_auto(
            chat_id=chat_id,
            message=query_to_use,
            rewrite_result=rewrite_result,
        )

    async def stream(
        self,
        session_id:        str,
        chat_id:           Optional[str],
        message:           str,
        model_preference:  str = "auto",
        rewriter_enabled:  bool = True,
    ):
        """
        Streaming version of run(). Yields SSE-compatible event dicts.

        Event types:
          {"type": "token",    "content": "<text chunk>"}
          {"type": "metadata", "chat_id": ..., "categories_used": [...], ...}
          {"type": "fallback"} — signals frontend to use non-streaming path (compound query)
          {"type": "error",    "detail": "<message>"}

        The final metadata event is emitted after all tokens so the frontend
        can attach badges (model used, category, savings) at the right moment.
        """
        # ── Step 1: Resolve or create chat + save user message ─────────────────
        if chat_id is None:
            chat_id = generate_id()
            title   = self._make_title(message)
            await queries.save_chat(chat_id, session_id, title)
            logger.info("Stream: Created new chat %s for session %s", chat_id, session_id)

        user_message_id  = generate_id()
        user_token_count = count_tokens(message)
        await queries.save_message(
            message_id=user_message_id,
            chat_id=chat_id,
            role="user",
            content=message,
            token_count=user_token_count,
        )

        # ── Rewriter ─────────────────────────────────────────────────────────────
        if rewriter_enabled:
            rewrite_result = await _rewriter.rewrite(message)
            query_to_use = rewrite_result.rewritten_query
        else:
            rewrite_result = None
            query_to_use = message

        # ── Manual mode: stream the chosen model directly ─────────────────────
        if model_preference != "auto":
            fallbacks = [m for m in ALL_MODELS_BY_QUALITY if m != model_preference]
            prompt = await _memory.build_prompt(chat_id=chat_id, current_query=query_to_use)
            full_response = ""
            model_used = model_preference

            async for chunk, used_model in _llm_client.async_stream(
                model=model_preference,
                messages=prompt,
                fallback_models=fallbacks,
                max_tokens=2048,
            ):
                if chunk:
                    full_response += chunk
                    model_used = used_model
                    yield {"type": "token", "content": chunk}
                else:
                    model_used = used_model  # sentinel — capture final model

            yield await self._finalize_stream(
                chat_id=chat_id,
                full_response=full_response,
                model_used=model_used,
                route_category="manual",
                categories_used=["manual"],
                models_used=[model_used],
                rewrite_result=rewrite_result,
                original_message=message,
            )
            return

        # ── Auto mode: classify ────────────────────────────────────────────────
        sub_queries = await _router.classify_and_split(query_to_use)
        logger.info("Stream Router: %d sub-task(s): %s", len(sub_queries), [s["category"] for s in sub_queries])

        # ── Compound query: signal fallback ───────────────────────────────────
        if len(sub_queries) > 1:
            logger.info("Stream: compound query — signalling fallback to non-streaming")
            yield {"type": "fallback", "chat_id": chat_id}
            return

        # ── Single intent: stream ─────────────────────────────────────────────
        sq = sub_queries[0]
        category  = sq["category"]
        sub_query = sq["sub_query"]
        full_response = ""
        model_used = "unknown"

        if category == "dsa":
            async for chunk, used_model in _dsa_agent.stream_solve(sub_query):
                if chunk:
                    full_response += chunk
                    model_used = used_model
                    yield {"type": "token", "content": chunk}
                else:
                    model_used = used_model
            model_used = f"subagent/dsa ({model_used})"

        elif category == "web_search":
            async for chunk, used_model in _web_searcher.stream_solve(sub_query):
                if chunk:
                    full_response += chunk
                    model_used = used_model
                    yield {"type": "token", "content": chunk}
                else:
                    model_used = used_model
            model_used = f"subagent/web_search ({model_used})"

        elif category == "evaluate":
            # Evaluator makes parallel judge calls first, then we stream the report
            report = await _evaluator.evaluate_pair(sub_query)
            # Stream the report text in chunks for a nice visual effect
            chunk_size = 8
            for i in range(0, len(report), chunk_size):
                piece = report[i:i + chunk_size]
                full_response += piece
                yield {"type": "token", "content": piece}
            model_used = "subagent/evaluate"

        else:
            # Direct LLM route
            models = ROUTE_MAP.get(category, ROUTE_MAP["general"])
            primary   = models["primary"]
            fallbacks = [models["fallback_1"], models["fallback_2"]]
            prompt = await _memory.build_prompt(chat_id=chat_id, current_query=sub_query)

            async for chunk, used_model in _llm_client.async_stream(
                model=primary,
                messages=prompt,
                fallback_models=fallbacks,
                max_tokens=2048,
            ):
                if chunk:
                    full_response += chunk
                    model_used = used_model
                    yield {"type": "token", "content": chunk}
                else:
                    model_used = used_model

        # ── Emit metadata + save to DB ────────────────────────────────────────
        yield await self._finalize_stream(
            chat_id=chat_id,
            full_response=full_response,
            model_used=model_used,
            route_category=category,
            categories_used=[category],
            models_used=[model_used],
            rewrite_result=rewrite_result,
            original_message=message,
        )

    async def _finalize_stream(
        self,
        chat_id: str,
        full_response: str,
        model_used: str,
        route_category: str,
        categories_used: List[str],
        models_used: List[str],
        rewrite_result,
        original_message: str,
    ) -> dict:
        """
        Save the completed streamed response to DB and return the metadata event dict.
        """
        assistant_message_id = generate_id()
        await queries.save_message(
            message_id=assistant_message_id,
            chat_id=chat_id,
            role="assistant",
            content=full_response,
            token_count=count_tokens(full_response),
            model_used=model_used,
            route_category=route_category,
            models_used=models_used,
            original_tokens=rewrite_result.original_tokens if rewrite_result else count_tokens(original_message),
            rewritten_tokens=rewrite_result.rewritten_tokens if rewrite_result else count_tokens(original_message),
            reduction_pct=rewrite_result.reduction_pct if rewrite_result else 0.0,
        )
        logger.info("Stream: saved response %s | category=%s | model=%s", assistant_message_id, route_category, model_used)

        return {
            "type":             "metadata",
            "chat_id":          chat_id,
            "message_id":       assistant_message_id,
            "model_used":       model_used,
            "route_category":   route_category,
            "categories_used":  categories_used,
            "models_used":      models_used,
            "original_tokens":  rewrite_result.original_tokens if rewrite_result else count_tokens(original_message),
            "rewritten_tokens": rewrite_result.rewritten_tokens if rewrite_result else count_tokens(original_message),
            "reduction_pct":    rewrite_result.reduction_pct if rewrite_result else 0.0,
        }



    # ─────────────────────────────────────────────────────────────────────────
    # MANUAL MODE
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_manual(
        self,
        chat_id: str,
        message: str,
        model:   str,
        rewrite_result: Optional[RewriteResult] = None,
    ) -> OrchestratorResult:
        """
        User explicitly chose a model — skip all routing, splitting, and judging.
        Build the prompt with conversation history and call the chosen model.

        Fallback behavior: if the chosen model fails, we try ALL other available
        models in quality order (best-to-cheapest). The user always sees which model
        ACTUALLY responded via model_used in the response — no silent surprises.

        Note: If rewriter_enabled=True, the query was already rewritten before
        reaching this method (Phase 2 will handle that in the run() method above).

        Args:
            chat_id: UUID4 of the conversation.
            message: The user's input (raw in Phase 3, rewritten in Phase 2+).
            model:   The LiteLLM model string chosen by the user.
        Returns:
            OrchestratorResult with route_category="manual".
        """
        logger.info("Manual mode: calling %s (with full fallback chain)", model)

        # Build fallback list: all models EXCEPT the one already being tried as primary
        # This prevents calling the same model twice if it fails
        fallbacks = [m for m in ALL_MODELS_BY_QUALITY if m != model]

        # Build prompt: system + last-3-history + current message
        prompt = await _memory.build_prompt(chat_id=chat_id, current_query=message)

        # Gemini 3+ models need temperature=1.0 to avoid infinite loop warnings
        temperature = _get_temperature(model)

        result = await _llm_client.async_complete(
            model=model,
            messages=prompt,
            fallback_models=fallbacks,  # full quality-ordered fallback chain
            temperature=temperature,
            max_tokens=2048,
        )

        # Save assistant response to DB
        assistant_message_id = generate_id()
        await queries.save_message(
            message_id=assistant_message_id,
            chat_id=chat_id,
            role="assistant",
            content=result.content,
            token_count=count_tokens(result.content, model=result.model_used),
            model_used=result.model_used,
            route_category="manual",
            models_used=[result.model_used],
            original_tokens=rewrite_result.original_tokens if rewrite_result else count_tokens(message),
            rewritten_tokens=rewrite_result.rewritten_tokens if rewrite_result else count_tokens(message),
            reduction_pct=rewrite_result.reduction_pct if rewrite_result else 0.0,
        )

        return OrchestratorResult(
            response=result.content,
            model_used=result.model_used,
            chat_id=chat_id,
            message_id=assistant_message_id,
            route_category="manual",
            categories_used=["manual"],
            models_used=[result.model_used],
            original_tokens=rewrite_result.original_tokens if rewrite_result else count_tokens(message),
            rewritten_tokens=rewrite_result.rewritten_tokens if rewrite_result else count_tokens(message),
            reduction_pct=rewrite_result.reduction_pct if rewrite_result else 0.0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # AUTO MODE
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_auto(
        self,
        chat_id: str,
        message: str,
        rewrite_result: Optional[RewriteResult] = None,
    ) -> OrchestratorResult:
        """
        Auto mode pipeline:
          1. Router classifies query → list of {category, sub_query} dicts
          2. Dispatch single or compound
          3. Judge each response
          4. Retry severity-3 failures
          5. Aggregate if compound, return directly if single

        Args:
            chat_id: UUID4 of the conversation.
            message: The user's raw input (Phase 3 — no rewriter yet).
        Returns:
            OrchestratorResult with full routing metadata.
        """
        # ── Classify + split ────────────────────────────────────────────────
        sub_queries = await _router.classify_and_split(message)
        logger.info(
            "Router produced %d sub-task(s): %s",
            len(sub_queries), [s["category"] for s in sub_queries]
        )

        # ── Dispatch all sub-queries in parallel ────────────────────────────
        # asyncio.gather runs all dispatch coroutines concurrently.
        # Each call returns (response_text, model_used_string).
        dispatch_tasks = [
            self._dispatch(sq["sub_query"], sq["category"], chat_id)
            for sq in sub_queries
        ]
        dispatch_results: List[Tuple[str, str]] = await asyncio.gather(*dispatch_tasks)

        # ── Judge all responses in parallel ─────────────────────────────────
        judge_tasks = [
            self._judge(sq["sub_query"], response)
            for sq, (response, _) in zip(sub_queries, dispatch_results)
        ]
        judgments = await asyncio.gather(*judge_tasks)

        # ── Retry any severity-3 failures ───────────────────────────────────
        # Build the final list of (response, model_used) after retries.
        final_results: List[Tuple[str, str]] = []

        for i, (judgment, (response, model_used)) in enumerate(zip(judgments, dispatch_results)):
            sq = sub_queries[i]

            if judgment["severity"] == 3:
                logger.warning(
                    "Severity 3 for category '%s': %s — triggering resplit+retry",
                    sq["category"], judgment["reason"]
                )
                # Resplit the sub-query and call the model once more
                response, model_used = await self._resplit_and_retry(
                    original_query=message,
                    failed_sub_query=sq["sub_query"],
                    failure_reason=judgment["reason"],
                    category=sq["category"],
                    chat_id=chat_id,
                )
            else:
                logger.debug(
                    "Severity %d for category '%s' — accepted immediately",
                    judgment["severity"], sq["category"]
                )

            final_results.append((response, model_used))

        # ── Single intent: return directly (no aggregation) ─────────────────
        if len(sub_queries) == 1:
            final_response = final_results[0][0]
            final_model    = final_results[0][1]
            route_category = sub_queries[0]["category"]
            categories_used = [route_category]
            models_used     = [final_model]

        # ── Compound intent: aggregate all responses into one reply ──────────
        else:
            sub_responses = [
                {
                    "category":  sub_queries[i]["category"],
                    "sub_query": sub_queries[i]["sub_query"],
                    "response":  final_results[i][0],
                }
                for i in range(len(sub_queries))
            ]
            final_response  = await self._aggregate(message, sub_responses)
            categories_used = [sq["category"] for sq in sub_queries]
            models_used     = [r[1] for r in final_results]
            # Compound route_category is stored as comma-separated string (Option B)
            route_category  = ",".join(categories_used)
            final_model     = "aggregated"

        # ── Save assistant response to DB ────────────────────────────────────
        assistant_message_id = generate_id()
        await queries.save_message(
            message_id=assistant_message_id,
            chat_id=chat_id,
            role="assistant",
            content=final_response,
            token_count=count_tokens(final_response),
            model_used=final_model,
            route_category=route_category,
            models_used=models_used,
            original_tokens=rewrite_result.original_tokens if rewrite_result else count_tokens(message),
            rewritten_tokens=rewrite_result.rewritten_tokens if rewrite_result else count_tokens(message),
            reduction_pct=rewrite_result.reduction_pct if rewrite_result else 0.0,
        )

        logger.info(
            "Pipeline complete | categories=%s | models=%s | route=%s",
            categories_used, models_used, route_category
        )

        return OrchestratorResult(
            response=final_response,
            model_used=final_model,
            chat_id=chat_id,
            message_id=assistant_message_id,
            route_category=route_category,
            categories_used=categories_used,
            models_used=models_used,
            original_tokens=rewrite_result.original_tokens if rewrite_result else count_tokens(message),
            rewritten_tokens=rewrite_result.rewritten_tokens if rewrite_result else count_tokens(message),
            reduction_pct=rewrite_result.reduction_pct if rewrite_result else 0.0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    async def _dispatch(
        self,
        sub_query: str,
        category:  str,
        chat_id:   str,
    ) -> Tuple[str, str]:
        """
        Call the correct specialized model for a single sub-query.

        Picks the primary model from ROUTE_MAP[category], with two automatic fallbacks.
        Builds the full prompt including conversation history via MemoryInjector.

        Args:
            sub_query: The specific sub-task text to answer.
            category:  One of the 7 route categories — used to pick the model.
            chat_id:   UUID4 — needed to fetch conversation history from DB.
        Returns:
            Tuple[str, str]: (response_text, model_used_string)
        """
        # ── Phase 5 Subagent Hook ──
        if category == "dsa":
            logger.info("Dispatching to DSASubagent for category 'dsa'...")
            response = await _dsa_agent.solve(sub_query)
            return response, "subagent/dsa"

        # ── Web Search Subagent Hook ──
        if category == "web_search":
            logger.info("Dispatching to WebSearchSubagent for category 'web_search'...")
            response = await _web_searcher.solve(sub_query)
            return response, "subagent/web_search"

        # ── Phase 6 Evaluator Hook ──
        if category == "evaluate":
            logger.info("Dispatching to EvaluatorSubagent for category 'evaluate'...")
            response = await _evaluator.evaluate_pair(sub_query)
            return response, "subagent/evaluate"

        # Pick the model chain for this category
        models = ROUTE_MAP.get(category, ROUTE_MAP["general"])
        primary   = models["primary"]
        fallbacks = [models["fallback_1"], models["fallback_2"]]

        # Build prompt: system + history + sub_query
        # Note: sub_query (not the original full message) is placed as the current user turn.
        # The model only needs to answer this specific sub-task, not the full compound query.
        prompt = await _memory.build_prompt(chat_id=chat_id, current_query=sub_query)

        temperature = _get_temperature(primary)

        result = await _llm_client.async_complete(
            model=primary,
            messages=prompt,
            fallback_models=fallbacks,
            temperature=temperature,
            max_tokens=2048,
        )

        logger.info(
            "Dispatch [%s] → %s | %d tokens out",
            category, result.model_used, result.completion_tokens
        )
        return result.content, result.model_used

    async def _judge(self, sub_query: str, response: str) -> dict:
        """
        Ask the cheap utility LLM to rate the quality of a model response.

        Severity scale:
          0 = Perfect — fully addresses the sub-task
          1 = Minor issue — correct but could be more complete
          2 = Moderate gap — partially addresses the sub-task
          3 = Severe failure — off-topic, empty, or factually broken

        Only severity 3 triggers a retry. 0, 1, 2 are accepted immediately.

        Args:
            sub_query: The specific sub-task that was sent to the model.
            response:  The model's response to that sub-task.
        Returns:
            dict with keys "severity" (int 0-3) and "reason" (str).
            On parse failure, returns {"severity": 0, "reason": "parse failed — accepted"}.
        """
        judge_prompt = f"""\
You are a response quality judge.

Sub-task that was asked: {sub_query}
Model response: {response}

Score this response from 0 to 3:
  0 = Perfect — fully and correctly addresses the sub-task
  1 = Minor issue — correct but slightly incomplete
  2 = Moderate gap — partially addresses the sub-task
  3 = Severe failure — off-topic, completely empty, or factually broken

Respond ONLY with a JSON object, no explanation:
{{"severity": <0|1|2|3>, "reason": "<one sentence>"}}"""

        messages = [{"role": "user", "content": judge_prompt}]

        try:
            result = await _llm_client.async_complete(
                model=UTILITY_PRIMARY,
                messages=messages,
                fallback_models=UTILITY_FALLBACKS,  # full chain — all models as backup
                temperature=0.0,   # deterministic judgment
                max_tokens=128,    # judgment response is always short
            )
            raw = result.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw[raw.find("\n") + 1:]
            if raw.endswith("```"):
                raw = raw[:-3]

            parsed = json.loads(raw.strip())

            severity = int(parsed.get("severity", 0))
            # Clamp to valid range in case LLM returns something unexpected
            severity = max(0, min(3, severity))
            reason   = str(parsed.get("reason", ""))

            logger.debug("Judgment: severity=%d | reason=%s", severity, reason)
            return {"severity": severity, "reason": reason}

        except Exception as e:
            # If judgment itself fails, treat as severity 0 (accept) — don't retry on judge failure
            logger.warning("Judgment failed (%s) — defaulting to severity 0 (accept)", e)
            return {"severity": 0, "reason": "judgment parse failed — accepted as-is"}

    async def _resplit_and_retry(
        self,
        original_query:  str,
        failed_sub_query: str,
        failure_reason:  str,
        category:        str,
        chat_id:         str,
    ) -> Tuple[str, str]:
        """
        Two-step corrected retry for severity-3 failures:
          1. Ask the utility LLM to write a better-scoped version of the failed sub-query.
          2. Call the same expensive specialized model once more with the corrected sub-query.
          The result of this retry is ALWAYS accepted — no second judgment.

        Hard ceiling: this is the max 2nd call to the expensive model for this sub-task.

        Args:
            original_query:   The full original user message (context for resplit).
            failed_sub_query: The sub-query that produced a severity-3 response.
            failure_reason:   The judge's explanation of what was wrong.
            category:         The route category — used to pick the right model again.
            chat_id:          UUID4 — needed to fetch conversation history.
        Returns:
            Tuple[str, str]: (response_text, model_used_string) — always accepted.
        """
        # ── Step 1: Ask utility LLM to produce a better sub-query ───────────
        resplit_prompt = f"""\
A specialized AI model failed to answer a sub-task properly.

Full original user query (context): {original_query}
Failed sub-task that was sent to the model: {failed_sub_query}
Why it failed (from judge): {failure_reason}

Write an improved, more specific version of the sub-task that gives the model
a better chance of answering correctly. Be concise and self-contained.
Respond ONLY with the improved sub-task text. No explanation. No JSON."""

        messages = [{"role": "user", "content": resplit_prompt}]

        try:
            resplit_result = await _llm_client.async_complete(
                model=UTILITY_PRIMARY,
                messages=messages,
                fallback_models=UTILITY_FALLBACKS,  # full chain — all models as backup
                temperature=0.3,   # slight creativity needed to produce a better sub-query
                max_tokens=256,
            )
            corrected_sub_query = resplit_result.content.strip()
            logger.info("Resplit produced corrected sub-query: %s", corrected_sub_query[:80])

        except Exception as e:
            # If resplit itself fails, fall back to the original sub-query
            logger.warning("Resplit LLM call failed (%s) — retrying with original sub-query", e)
            corrected_sub_query = failed_sub_query

        # ── Step 2: Call the expensive model once more with corrected sub-query
        # This is the FINAL call for this sub-task — result is accepted unconditionally.
        response, model_used = await self._dispatch(
            sub_query=corrected_sub_query,
            category=category,
            chat_id=chat_id,
        )
        logger.info("Retry complete for category '%s' — result accepted as final", category)
        return response, model_used

    async def _aggregate(
        self,
        original_query: str,
        sub_responses:  List[dict],
    ) -> str:
        """
        Ask the cheap utility LLM to synthesize multiple sub-responses into one
        coherent, well-organized final reply.

        Called ONLY for compound queries (2+ sub-tasks). Never called for single-intent.

        Args:
            original_query: The full original user message — used to frame the synthesis.
            sub_responses:  List of dicts, each with "category", "sub_query", "response".
        Returns:
            str: A single coherent reply combining all sub-answers.
        """
        # Build a formatted block of all sub-responses for the aggregation prompt
        sub_blocks = "\n\n".join(
            f"--- [Category: {sr['category']}] ---\n"
            f"Sub-task: {sr['sub_query']}\n"
            f"Answer: {sr['response']}"
            for sr in sub_responses
        )

        aggregation_prompt = f"""\
You are an expert assistant synthesizing multiple AI responses into one final answer.

Original user question: {original_query}

The following sub-answers were generated by specialized models:
{sub_blocks}

Write a single, coherent, well-organized response that:
1. Addresses all parts of the original question completely.
2. Integrates all sub-answers naturally — do not just list them separately.
3. If any sub-answers conflict with each other, explicitly note the conflict to the user.
4. Uses clear headings or sections if the answer covers multiple distinct topics.

Write the final combined answer now:"""

        messages = [{"role": "user", "content": aggregation_prompt}]

        try:
            result = await _llm_client.async_complete(
                model=UTILITY_PRIMARY,
                messages=messages,
                fallback_models=UTILITY_FALLBACKS,  # full chain — all models as backup
                temperature=0.7,   # some creativity for smooth synthesis
                max_tokens=2048,
            )
            logger.info("Aggregation complete | %d tokens out", result.completion_tokens)
            return result.content

        except Exception as e:
            # If aggregation fails, fall back to concatenating the sub-responses directly
            logger.error("Aggregation LLM call failed (%s) — falling back to concat", e)
            return "\n\n---\n\n".join(
                f"**{sr['category'].upper()}**\n{sr['response']}"
                for sr in sub_responses
            )

    @staticmethod
    def _make_title(message: str) -> str:
        """
        Generate a short sidebar title from the user's first message.
        Takes the first 6 words, appends '…' if the message is longer.
        """
        words = message.strip().split()
        if len(words) <= 6:
            return " ".join(words)
        return " ".join(words[:6]) + "…"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_temperature(model: str) -> float:
    """
    Return the correct temperature for a given model.

    Gemini 3+ models (gemini-3.5-flash etc.) produce infinite loops and degraded
    reasoning when temperature < 1.0 — LiteLLM itself warns about this.
    All other models (Groq-hosted) work best with 0.7.

    Args:
        model: LiteLLM model string (e.g. "gemini/gemini-3.5-flash")
    Returns:
        float: 1.0 for Gemini, 0.7 for everything else.
    """
    return 1.0 if "gemini" in model else 0.7
