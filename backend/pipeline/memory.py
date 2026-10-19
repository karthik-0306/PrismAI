"""
backend/pipeline/memory.py

Builds the conversation history injected into every LLM call, and keeps it
bounded with rolling summarization.

How a prompt is assembled:
  [system: persona + today's date + rolling summary of older turns (if any)]
  [recent turns verbatim, newest-first until HISTORY_TOKEN_BUDGET is spent]
  [current user query]

Invariants (each of these was a real bug before):
  - History is loaded BEFORE the current message is saved, so the current query
    is never sent twice (models answered "and times 3?" by multiplying twice).
  - Empty assistant rows and orphaned user turns (a question whose answer
    failed) are dropped, so roles always alternate user → assistant.
  - Summarization actually runs (it existed but was never called), compresses
    only the OLDER messages and always keeps the last few turns verbatim.
"""

import asyncio
import logging
import uuid
from datetime import date
from typing import Dict, List, Optional

from backend import config
from backend.database import queries
from backend.database.models import Message
from backend.llm.client import LLMClient
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are PrismAI, a helpful, knowledgeable AI assistant. "
    "Today's date is {today}. "
    "Use the conversation so far to resolve follow-up questions (\"it\", \"that\", "
    "\"the previous one\") without asking the user to repeat themselves. "
    "Answer directly and accurately. Use Markdown (headings, lists, tables, fenced "
    "code blocks with a language tag) when it improves readability; keep simple "
    "answers short. If you are unsure or something may be outdated, say so."
)

SUMMARIZE_SYSTEM_PROMPT = (
    "You compress conversation history into a dense summary for an AI assistant's memory.\n"
    "- Merge the OLD SUMMARY (if any) with the NEW MESSAGES into ONE updated summary.\n"
    "- Keep facts about the user (name, preferences, goals), topics covered, decisions, "
    "key results, code/function names and open questions.\n"
    "- Drop pleasantries and intermediate reasoning.\n"
    "- Output only the summary text, at most ~250 words."
)

# Per-message cap inside history so one giant paste can't evict everything else.
_MAX_TOKENS_PER_HISTORY_MESSAGE = 1200

_summarize_locks: Dict[str, asyncio.Lock] = {}


def _truncate_middle(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    max_chars = max_tokens * 4
    head, tail = text[: int(max_chars * 0.7)], text[-int(max_chars * 0.3):]
    return f"{head}\n\n[… earlier content trimmed …]\n\n{tail}"


def _clean_turns(messages: List[Message]) -> List[Message]:
    """Drop empty rows and unanswered user turns; guarantee user/assistant alternation."""
    cleaned: List[Message] = []
    for msg in messages:
        if not (msg.content or "").strip():
            continue
        if cleaned and cleaned[-1].role == msg.role:
            if msg.role == "user":
                cleaned[-1] = msg   # previous question never got an answer — drop it
            # (a duplicate assistant reply is skipped — keep the first)
            continue
        cleaned.append(msg)
    if cleaned and cleaned[-1].role == "user":
        cleaned.pop()               # trailing unanswered question
    while cleaned and cleaned[0].role == "assistant":
        cleaned.pop(0)              # history must start with a user turn
    return cleaned


class MemoryInjector:
    """Stateless helper — safe to share across concurrent requests."""

    def __init__(self):
        self._llm = LLMClient()

    async def load_history(self, chat_id: Optional[str]) -> List[dict]:
        """
        Return the prior conversation as LiteLLM messages (oldest first), including
        a leading system message carrying the rolling summary when one exists.
        Must be called before the current user message is saved.
        """
        if not chat_id:
            return []

        summary = await queries.get_latest_summary(chat_id)
        turns = _clean_turns(await queries.get_unsummarized_messages(chat_id))

        budget = config.HISTORY_TOKEN_BUDGET
        picked: List[dict] = []
        for msg in reversed(turns):
            content = _truncate_middle(msg.content, _MAX_TOKENS_PER_HISTORY_MESSAGE)
            cost = count_tokens(content)
            if picked and cost > budget:
                break
            budget -= cost
            picked.append({"role": msg.role, "content": content})
        picked.reverse()
        # Keep pairs intact: never start on an assistant reply.
        while picked and picked[0]["role"] == "assistant":
            picked.pop(0)

        history: List[dict] = []
        if summary:
            history.append({
                "role": "system",
                "content": f"Summary of the earlier part of this conversation:\n{summary.content}",
            })
        history.extend(picked)
        return history

    @staticmethod
    def build_prompt(history: List[dict], current_query: str,
                     system_prompt: Optional[str] = None) -> List[dict]:
        """
        Merge persona, rolling summary, recent turns and the current query into one
        messages list. The summary is folded into the single system message because
        some providers reject or reorder multiple system messages.
        """
        system = (system_prompt or SYSTEM_PROMPT).replace("{today}", date.today().strftime("%A, %B %d, %Y"))
        turns = []
        for msg in history:
            if msg["role"] == "system":
                system += "\n\n" + msg["content"]
            else:
                turns.append(msg)
        return [{"role": "system", "content": system}, *turns, {"role": "user", "content": current_query}]

    @staticmethod
    def context_snippet(history: List[dict], max_messages: int = 4, max_chars: int = 500) -> str:
        """Compact transcript of the last few turns, for the router/rewriter."""
        lines = []
        for msg in history:
            if msg["role"] == "system":
                lines.append(f"[Earlier summary]: {msg['content'][:max_chars]}")
        for msg in [m for m in history if m["role"] != "system"][-max_messages:]:
            text = " ".join(msg["content"].split())
            if len(text) > max_chars:
                text = text[:max_chars] + "…"
            lines.append(f"{'User' if msg['role'] == 'user' else 'Assistant'}: {text}")
        return "\n".join(lines)

    async def maybe_summarize(self, chat_id: str) -> None:
        """
        Compress older turns into the rolling summary when unsummarized history
        exceeds SUMMARIZE_TRIGGER_TOKENS. Safe to fire-and-forget; never raises.
        """
        lock = _summarize_locks.setdefault(chat_id, asyncio.Lock())
        if lock.locked():
            return
        async with lock:
            try:
                await self._summarize(chat_id)
            except Exception as e:
                logger.warning("Memory: summarization failed for chat %s: %s", chat_id, e)
            finally:
                _summarize_locks.pop(chat_id, None)

    async def _summarize(self, chat_id: str) -> None:
        unsummarized = await queries.get_unsummarized_messages(chat_id)
        total = sum(m.token_count or count_tokens(m.content) for m in unsummarized)
        if total <= config.SUMMARIZE_TRIGGER_TOKENS or len(unsummarized) <= config.KEEP_RECENT_MESSAGES:
            return

        to_compress = unsummarized[: -config.KEEP_RECENT_MESSAGES]
        old_summary = await queries.get_latest_summary(chat_id)

        body = ""
        if old_summary:
            body += f"<OLD_SUMMARY>\n{old_summary.content}\n</OLD_SUMMARY>\n\n"
        body += "<NEW_MESSAGES>\n"
        for msg in to_compress:
            label = "User" if msg.role == "user" else "Assistant"
            body += f"[{label}]: {_truncate_middle(msg.content, 800)}\n\n"
        body += "</NEW_MESSAGES>"

        logger.info("Memory: summarizing %d messages (%d tokens) in chat %s", len(to_compress), total, chat_id)
        result = await self._llm.complete(
            models=config.UTILITY_CHAIN,
            messages=[{"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                      {"role": "user", "content": body}],
            temperature=0.3,
            max_tokens=600,
            low_reasoning=True,
        )

        await queries.save_summary(
            summary_id=str(uuid.uuid4()),
            chat_id=chat_id,
            content=result.content,
            covers_up_to=to_compress[-1].message_id,
        )
        await queries.mark_messages_summarized([m.message_id for m in to_compress])
        logger.info("Memory: chat %s compressed %d messages into a summary", chat_id, len(to_compress))
