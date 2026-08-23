"""
backend/pipeline/memory.py

Builds the LLM prompt by injecting conversation history.

Phase 4 version: continuous memory compression.
As conversations grow, this module squashes older messages into a dense summary
while keeping recent messages verbatim. This prevents the context window from
ever overflowing, saving tokens and money while retaining memory.

How the prompt is assembled:
  [system message]
  [latest memory summary (if exists)]
  [last N unsummarized messages verbatim]
  [current user query]
"""

import logging
import uuid
from typing import List

from backend.database import queries  # DB read/write functions
from backend.database.models import Message, Summary
from backend.llm.client import LLMClient
from litellm import model_cost

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# How many recent messages to include verbatim AFTER the summary.
# 6 messages = 3 full exchanges (user + assistant).
VERBATIM_HISTORY_COUNT = 6

# Percentage of the max context window at which summarization is triggered.
CONTEXT_THRESHOLD_PCT = 0.60

# Default max tokens if litellm doesn't know the model (rare).
FALLBACK_MAX_TOKENS = 8192

# The fast, cheap model used to compress the history in the background.
SUMMARY_MODEL = "gemini/gemini-3.5-flash"

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are PrismAI, an expert AI assistant. "
    "You have access to the recent conversation history below. "
    "Always maintain context from prior messages. "
    "Be concise, accurate, and helpful."
)

SUMMARIZE_SYSTEM_PROMPT = (
    "You are a MEMORY COMPRESSOR. Your job is to compress a conversation history into a dense, "
    "comprehensive summary paragraph.\n\n"
    "RULES:\n"
    "1. If an OLD SUMMARY is provided, merge its facts with the NEW MESSAGES to create ONE updated summary.\n"
    "2. Keep all core facts, user preferences, names, concepts discussed, and decisions made.\n"
    "3. Discard filler words, pleasantries, and step-by-step reasoning that isn't the final answer.\n"
    "4. Output ONLY the summary text. No intro, no XML, no meta-commentary."
)


class MemoryInjector:
    """
    Builds the messages list (prompt) for each LLM call and manages history compression.
    """
    def __init__(self):
        self._llm = LLMClient()

    async def build_prompt(self, chat_id: str, current_query: str) -> List[dict]:
        """
        Construct the full messages list to send to the LLM.

        Args:
            chat_id:       UUID4 string identifying the active conversation.
            current_query: The user's latest input (not yet saved to DB).
        Returns:
            List[dict]: ordered messages list ready to pass to litellm.acompletion.
        """
        prompt: List[dict] = []
        prompt.append({"role": "system", "content": SYSTEM_PROMPT})

        # ── 1. Inject the latest summary (if any) ────────────────────────────
        summary = await queries.get_latest_summary(chat_id)
        if summary:
            logger.debug("Memory: injecting summary %s", summary.summary_id)
            prompt.append({
                "role": "system",
                "content": f"<OLD_MEMORY_SUMMARY>\n{summary.content}\n</OLD_MEMORY_SUMMARY>"
            })

        # ── 2. Fetch unsummarized history from DB ────────────────────────────
        # Only inject messages that haven't been compressed into the summary yet
        unsummarized: List[Message] = await queries.get_unsummarized_messages(chat_id)

        # Take only the last N messages
        recent: List[Message] = unsummarized[-VERBATIM_HISTORY_COUNT:]

        logger.debug(
            "Memory: Building prompt for chat %s: %d unsummarized history messages + 1 current",
            chat_id, len(recent)
        )

        # ── 3. Inject recent history ─────────────────────────────────────────
        for msg in recent:
            prompt.append({"role": msg.role, "content": msg.content})

        # ── 4. Inject current query ──────────────────────────────────────────
        prompt.append({"role": "user", "content": current_query})

        return prompt

    async def should_summarize(self, chat_id: str, model: str) -> bool:
        """
        Check if the unsummarized messages have exceeded our context budget.

        Args:
            chat_id: UUID4 string identifying the conversation.
            model:   LiteLLM model string to lookup the context window.
        Returns:
            bool: True if summarization should be triggered in the background.
        """
        unsummarized = await queries.get_unsummarized_messages(chat_id)
        
        # Don't bother summarizing if there's almost nothing there
        if len(unsummarized) < 3:
            return False

        # Calculate how many tokens the unsummarized messages consume
        total_tokens = sum(msg.token_count for msg in unsummarized)

        # Look up the model's context window size
        max_tokens = FALLBACK_MAX_TOKENS
        if model in model_cost:
            max_tokens = model_cost[model].get("max_input_tokens", FALLBACK_MAX_TOKENS)

        threshold = int(max_tokens * CONTEXT_THRESHOLD_PCT)

        logger.debug(
            "Memory: chat %s unsummarized tokens: %d / %d (threshold: %d)",
            chat_id, total_tokens, max_tokens, threshold
        )

        return total_tokens > threshold

    async def trigger_summarization(self, chat_id: str) -> None:
        """
        Compress all unsummarized messages into a new summary.
        Saves the summary to DB and marks the messages.
        Intended to be run asynchronously so it doesn't block the user's response.
        """
        logger.info("Memory: Triggering summarization for chat %s", chat_id)
        
        # 1. Gather materials
        unsummarized = await queries.get_unsummarized_messages(chat_id)
        if not unsummarized:
            logger.info("Memory: No messages to summarize for chat %s", chat_id)
            return

        old_summary = await queries.get_latest_summary(chat_id)
        
        # 2. Build the summarizer prompt
        prompt_content = ""
        if old_summary:
            prompt_content += f"<OLD_SUMMARY>\n{old_summary.content}\n</OLD_SUMMARY>\n\n"
        
        prompt_content += "<NEW_MESSAGES>\n"
        for msg in unsummarized:
            role_label = "User" if msg.role == "user" else "Assistant"
            prompt_content += f"[{role_label}]: {msg.content}\n\n"
        prompt_content += "</NEW_MESSAGES>"

        messages = [
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_content}
        ]

        # 3. Call utility LLM
        try:
            result = await self._llm.async_complete(
                model=SUMMARY_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1024
            )
            new_summary_text = result.content.strip()
            if not new_summary_text:
                raise ValueError("LLM returned empty summary")
        except Exception as e:
            logger.error("Memory: Summarization failed for chat %s: %s", chat_id, e)
            return

        # 4. Save results to DB
        summary_id = str(uuid.uuid4())
        last_message_id = unsummarized[-1].message_id
        
        await queries.save_summary(
            summary_id=summary_id,
            chat_id=chat_id,
            content=new_summary_text,
            covers_up_to=last_message_id
        )

        message_ids = [m.message_id for m in unsummarized]
        await queries.mark_messages_summarized(message_ids)

        logger.info(
            "Memory: Summarization complete. Compressed %d messages into summary %s", 
            len(message_ids), summary_id
        )
