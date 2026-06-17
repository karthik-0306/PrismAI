"""
backend/pipeline/orchestrator.py

The central coordinator for every user request.

The orchestrator is the ONLY file that knows the full pipeline sequence.
Every other module (memory, rewriter, router, LLM client, subagents) is isolated
and has no knowledge of what runs before or after it.

Phase 1 pipeline:
  1. Ensure chat exists in DB (create if new)
  2. Save user message to DB
  3. Build prompt with memory (last 3 messages)
  4. Call Gemini Flash (primary general model)
  5. Save assistant response to DB
  6. Return result to the router

Phases 2-6 will each add one step to this pipeline without touching other modules.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from backend.database import queries            # DB read/write operations
from backend.llm.client import LLMClient, get_fallbacks_for_route  # LLM abstraction
from backend.pipeline.memory import MemoryInjector               # prompt builder
from backend.utils.session import generate_id                    # UUID generator
from backend.utils.token_counter import count_tokens             # token estimation

logger = logging.getLogger(__name__)

# ── Module-level singletons — created once, reused on every request ───────────
# These are stateless — safe to share across concurrent async requests.
_llm_client = LLMClient()
_memory = MemoryInjector()


@dataclass
class OrchestratorResult:
    """
    Typed return value from Orchestrator.run().
    The chat router unpacks this and returns it as the HTTP response body.
    """
    response: str              # the assistant's reply text (think-tags stripped)
    model_used: str            # LiteLLM string of the model that produced the response
    chat_id: str               # UUID4 — the conversation this message belongs to
    message_id: str            # UUID4 — the assistant's message row in the DB
    route_category: str        # e.g. "general" — which ROUTE_MAP bucket was used
    eval_score: Optional[float] = None  # Phase 6: evaluator score (None until then)


class Orchestrator:
    """
    Coordinates the full request-response pipeline for one user message.

    Instantiated once per request (or reused as a singleton — stateless).
    All state lives in the DB and is re-fetched each turn.
    """

    async def run(
        self,
        session_id: str,
        chat_id: Optional[str],
        message: str,
        model_preference: str = "auto",
    ) -> OrchestratorResult:
        """
        Execute the Phase 1 pipeline for one user message.

        Args:
            session_id:       UUID4 string — the browser's session identifier.
            chat_id:          UUID4 string of an existing chat, or None to start a new one.
            message:          The user's raw input text.
            model_preference: "auto" uses ROUTE_MAP general; any other string is
                              a manual model override (Phase 3 feature, accepted here
                              but always uses auto in Phase 1).
        Returns:
            OrchestratorResult: all data the router needs to form the HTTP response.
        Raises:
            LLMError: propagated from LLMClient if all models fail.
        """
        # ── Step 1: Resolve or create the chat ──────────────────────────────
        is_new_chat = chat_id is None
        if is_new_chat:
            # Generate a fresh UUID4 for this new conversation thread
            chat_id = generate_id()
            # Use first 6 words of the user's message as the sidebar title
            title = self._make_title(message)
            await queries.save_chat(chat_id, session_id, title)
            logger.info("Created new chat %s for session %s", chat_id, session_id)

        # ── Step 2: Save the user's message to DB ───────────────────────────
        user_message_id = generate_id()
        user_token_count = count_tokens(message)

        await queries.save_message(
            message_id=user_message_id,
            chat_id=chat_id,
            role="user",
            content=message,
            token_count=user_token_count,
            # model_used and route_category are None for user messages
        )
        logger.debug("Saved user message %s (%d tokens)", user_message_id, user_token_count)

        # ── Step 3: Build the prompt with conversation history ───────────────
        # MemoryInjector fetches last 3 messages + prepends system message
        # current_query is appended last by build_prompt
        prompt = await _memory.build_prompt(chat_id=chat_id, current_query=message)

        # ── Step 4: Choose model and call the LLM ───────────────────────────
        # Phase 1: always use "general" route regardless of model_preference.
        # Phase 3 will implement the smart router and manual override.
        route_category = "general"
        primary_model, fallback_models = get_fallbacks_for_route(route_category)

        result = await _llm_client.async_complete(
            model=primary_model,
            messages=prompt,
            fallback_models=fallback_models,
            temperature=0.7,
            max_tokens=2048,
        )

        logger.info(
            "LLM response received from %s | %d tokens out",
            result.model_used, result.completion_tokens
        )

        # ── Step 5: Save the assistant's response to DB ─────────────────────
        assistant_message_id = generate_id()
        assistant_token_count = count_tokens(result.content, model=result.model_used)

        await queries.save_message(
            message_id=assistant_message_id,
            chat_id=chat_id,
            role="assistant",
            content=result.content,
            token_count=assistant_token_count,
            model_used=result.model_used,
            route_category=route_category,
        )
        logger.debug(
            "Saved assistant message %s (%d tokens)", assistant_message_id, assistant_token_count
        )

        # ── Step 6: Return structured result to the router ──────────────────
        return OrchestratorResult(
            response=result.content,
            model_used=result.model_used,
            chat_id=chat_id,
            message_id=assistant_message_id,
            route_category=route_category,
            eval_score=None,  # Phase 6 will populate this
        )

    @staticmethod
    def _make_title(message: str) -> str:
        """
        Generate a short sidebar title from the user's first message.
        Takes the first 6 words, joined by spaces. Truncates with "…" if longer.

        Args:
            message: the raw user input text.
        Returns:
            str: a title string of at most 6 words.
        """
        words = message.strip().split()
        if len(words) <= 6:
            return " ".join(words)
        return " ".join(words[:6]) + "…"
