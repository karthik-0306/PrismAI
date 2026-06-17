"""
backend/pipeline/memory.py

Builds the LLM prompt by injecting conversation history.

Phase 1 version: simple — fetches the last 3 messages verbatim.
No summarization yet (that's Phase 4).

How the prompt is assembled:
  [system message]
  [last 3 messages from DB as user/assistant turns]
  [current user query]

The system message sets the assistant's persona and tells it to use
prior context. Keeping it short saves tokens on every call.
"""

import logging
from typing import List

from backend.database import queries  # DB read functions
from backend.database.models import Message  # typed message model

logger = logging.getLogger(__name__)

# ── How many recent messages to include verbatim in the prompt ────────────────
# 3 messages = 1.5 full exchanges (user + assistant pairs).
# Enough to give continuity without burning context budget in Phase 1.
VERBATIM_HISTORY_COUNT = 3

# ── System prompt ─────────────────────────────────────────────────────────────
# Short and precise — tells the model its role and instructs it to reference
# the conversation history provided below it in the message list.
SYSTEM_PROMPT = (
    "You are PrismAI, an expert AI assistant. "
    "You have access to the recent conversation history below. "
    "Always maintain context from prior messages. "
    "Be concise, accurate, and helpful."
)


class MemoryInjector:
    """
    Builds the messages list (prompt) for each LLM call by injecting
    relevant conversation history from the database.

    Phase 1: fetches the last VERBATIM_HISTORY_COUNT messages only.
    Phase 4 will add: latest summary + last 3 verbatim + summarization trigger.
    """

    async def build_prompt(self, chat_id: str, current_query: str) -> List[dict]:
        """
        Construct the full messages list to send to the LLM.

        Structure returned:
            [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": "<oldest of last 3>"},
                {"role": "assistant", "content": "<response to that>"},
                ...
                {"role": "user",      "content": current_query},  ← always last
            ]

        Args:
            chat_id:       UUID4 string identifying the active conversation.
            current_query: The user's latest input (not yet saved to DB).
        Returns:
            List[dict]: ordered messages list ready to pass to litellm.acompletion.
        Side effects: reads from the messages table (read-only, no writes here).
        """
        # ── 1. Fetch history from DB ─────────────────────────────────────────
        all_messages: List[Message] = await queries.get_chat_messages(chat_id)

        # Take only the last N messages (oldest-first order is preserved by get_chat_messages)
        recent: List[Message] = all_messages[-VERBATIM_HISTORY_COUNT:]

        logger.debug(
            "Building prompt for chat %s: %d history messages + 1 current",
            chat_id, len(recent)
        )

        # ── 2. Assemble messages list ────────────────────────────────────────
        prompt: List[dict] = []

        # Always start with the system message
        prompt.append({"role": "system", "content": SYSTEM_PROMPT})

        # Inject recent history (user and assistant turns interleaved as they were stored)
        for msg in recent:
            prompt.append({"role": msg.role, "content": msg.content})

        # The current query is appended last — this is what the model will respond to
        prompt.append({"role": "user", "content": current_query})

        return prompt

    async def should_summarize(self, chat_id: str, model: str) -> bool:
        """
        Phase 1 stub — always returns False.
        Phase 4 will implement: calculate total token count and compare against
        60% of the model's context window. If exceeded, return True.

        Args:
            chat_id: UUID4 string identifying the conversation.
            model:   LiteLLM model string (needed for context window lookup in Phase 4).
        Returns:
            bool: always False in Phase 1.
        """
        # Phase 4 placeholder — do not remove this method, it is called by the orchestrator
        return False
