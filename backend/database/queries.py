"""
backend/database/queries.py

All SQL operations for PrismAI.

Rules enforced in this file:
  1. Every SQL statement uses parameterized queries (?, not f-strings) to prevent injection.
  2. All functions are async — they must be awaited by the caller.
  3. Functions return typed dataclasses (from models.py), not raw aiosqlite.Row objects.
  4. No business logic here — this file only reads/writes the DB.
     The orchestrator and pipeline modules handle logic.
"""

import json
import logging
from typing import Optional, List
from backend.database.connection import get_db       # the connection context manager
from backend.database.models import Chat, Message, Summary  # typed return types

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CHAT QUERIES
# ─────────────────────────────────────────────────────────────────────────────

async def save_chat(chat_id: str, session_id: str, title: str) -> None:
    """
    Insert a new chat row into the chats table.
    Called the first time a user sends a message in a new conversation.

    Args:
        chat_id:    UUID4 string identifying this conversation.
        session_id: UUID4 string identifying the user's browser session.
        title:      First 6 words of the first user message, used as sidebar label.
    Returns: None
    Side effects: Writes one row to chats.
    """
    async with get_db() as db:
        await db.execute(
            "INSERT INTO chats (chat_id, session_id, title) VALUES (?, ?, ?)",
            (chat_id, session_id, title)  # parameterized — no string formatting
        )
        await db.commit()
    logger.debug("Saved chat %s for session %s", chat_id, session_id)


async def get_all_chats_for_session(session_id: str) -> List[Chat]:
    """
    Retrieve all chats belonging to a browser session, ordered newest first.
    Used by GET /chats to populate the sidebar.

    Args:
        session_id: UUID4 string identifying the user's browser session.
    Returns:
        List[Chat]: all chat rows for this session, newest first.
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT chat_id, session_id, title, created_at "
            "FROM chats WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,)
        )
        rows = await cursor.fetchall()

    # Convert each raw DB row into a typed Chat dataclass
    return [
        Chat(
            chat_id=row["chat_id"],
            session_id=row["session_id"],
            title=row["title"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


async def search_chats(session_id: str, query: str) -> List[Chat]:
    """
    Search past conversations by keyword in the title or message content.
    Returns matched chats for the session, newest first.
    """
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT DISTINCT c.chat_id, c.session_id, c.title, c.created_at
            FROM chats c
            LEFT JOIN messages m ON c.chat_id = m.chat_id
            WHERE c.session_id = ? 
              AND (c.title LIKE ? OR m.content LIKE ?)
            ORDER BY c.created_at DESC
            """,
            (session_id, f"%{query}%", f"%{query}%")
        )
        rows = await cursor.fetchall()

    return [
        Chat(
            chat_id=row["chat_id"],
            session_id=row["session_id"],
            title=row["title"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


async def delete_chat(chat_id: str) -> None:
    """
    Delete a chat and all its associated messages from the database.
    """
    async with get_db() as db:
        await db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        await db.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
        await db.commit()
    logger.info("Deleted chat %s and all its messages", chat_id)


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE QUERIES
# ─────────────────────────────────────────────────────────────────────────────

async def save_message(
    message_id: str,
    chat_id: str,
    role: str,
    content: str,
    token_count: int,
    model_used: Optional[str] = None,
    route_category: Optional[str] = None,
    models_used: Optional[List[str]] = None,
    original_tokens: int = 0,
    rewritten_tokens: int = 0,
    reduction_pct: float = 0.0,
) -> None:
    """
    Insert a single message (user or assistant turn) into the messages table.
    Both the user's query and the LLM's reply are stored separately via this function.

    Args:
        message_id:     UUID4 string for this specific message.
        chat_id:        UUID4 string — which conversation this belongs to.
        role:           'user' or 'assistant'.
        content:        The text of the message.
        token_count:    Estimated token count (from token_counter.py).
        model_used:     LiteLLM model string (None for user messages).
                        For compound responses, set to 'aggregated'.
        route_category: Router classification result (None for user messages).
                        For compound responses, stored as CSV e.g. 'dsa,math'.
        models_used:    Full list of model strings that contributed to this response.
                        e.g. ['groq/openai/gpt-oss-120b', 'groq/qwen/qwen3-32b'].
                        Serialized to JSON for storage. None for user messages.
    Returns: None
    Side effects: Writes one row to messages.
    """
    # Serialize the models list to a compact JSON string for storage.
    # None → stored as NULL in the DB (user messages, pre-migration rows).
    models_json: Optional[str] = json.dumps(models_used) if models_used else None

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO messages
                (message_id, chat_id, role, content, model_used, route_category,
                 token_count, models_used_json, original_tokens, rewritten_tokens, reduction_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, chat_id, role, content, model_used, route_category,
             token_count, models_json, original_tokens, rewritten_tokens, reduction_pct)
        )
        await db.commit()
    logger.debug("Saved %s message %s in chat %s", role, message_id, chat_id)


async def get_chat_messages(chat_id: str) -> List[Message]:
    """
    Retrieve all messages for a chat, ordered oldest first.
    Used for two purposes:
      1. The history router returns this list to the frontend.
      2. The memory injector uses the last N rows to build the LLM prompt.

    Args:
        chat_id: UUID4 string identifying the conversation.
    Returns:
        List[Message]: all message rows for this chat, oldest first.
    """
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT message_id, chat_id, role, content, model_used,
                   route_category, token_count, is_summarized, created_at,
                   models_used_json, original_tokens, rewritten_tokens, reduction_pct
            FROM messages
            WHERE chat_id = ?
            ORDER BY created_at ASC
            """,
            (chat_id,)
        )
        rows = await cursor.fetchall()

    return [
        Message(
            message_id=row["message_id"],
            chat_id=row["chat_id"],
            role=row["role"],
            content=row["content"],
            model_used=row["model_used"],
            route_category=row["route_category"],
            token_count=row["token_count"],
            is_summarized=bool(row["is_summarized"]),  # SQLite stores 0/1, convert to bool
            created_at=row["created_at"],
            # Deserialize JSON list; fall back to None for legacy rows (column is NULL)
            models_used_json=row["models_used_json"],
            original_tokens=row["original_tokens"] if "original_tokens" in row.keys() else 0,
            rewritten_tokens=row["rewritten_tokens"] if "rewritten_tokens" in row.keys() else 0,
            reduction_pct=row["reduction_pct"] if "reduction_pct" in row.keys() else 0.0,
        )
        for row in rows
    ]


async def get_unsummarized_messages(chat_id: str) -> List[Message]:
    """
    Return only messages that have NOT yet been rolled into a summary.
    Used by the memory injector in Phase 4 to decide what to summarise next.

    Args:
        chat_id: UUID4 string identifying the conversation.
    Returns:
        List[Message]: messages with is_summarized = 0, oldest first.
    """
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT message_id, chat_id, role, content, model_used,
                   route_category, token_count, is_summarized, created_at,
                   models_used_json
            FROM messages
            WHERE chat_id = ? AND is_summarized = 0
            ORDER BY created_at ASC
            """,
            (chat_id,)
        )
        rows = await cursor.fetchall()

    return [
        Message(
            message_id=row["message_id"],
            chat_id=row["chat_id"],
            role=row["role"],
            content=row["content"],
            model_used=row["model_used"],
            route_category=row["route_category"],
            token_count=row["token_count"],
            is_summarized=bool(row["is_summarized"]),
            created_at=row["created_at"],
            models_used_json=row["models_used_json"],
        )
        for row in rows
    ]


async def mark_messages_summarized(message_ids: List[str]) -> None:
    """
    Mark a batch of messages as summarized so they are excluded from
    future verbatim context injection.
    Uses a single UPDATE with IN clause — one round-trip regardless of batch size.

    Args:
        message_ids: List of UUID4 strings to mark as summarized.
    Returns: None
    Side effects: Sets is_summarized = 1 on each specified message.
    """
    if not message_ids:
        return  # nothing to mark; avoid a no-op SQL call

    # Build parameterized placeholders: (?, ?, ?) for however many IDs
    placeholders = ", ".join("?" for _ in message_ids)
    async with get_db() as db:
        await db.execute(
            f"UPDATE messages SET is_summarized = 1 WHERE message_id IN ({placeholders})",
            tuple(message_ids)
        )
        await db.commit()
    logger.debug("Marked %d messages as summarized", len(message_ids))


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY QUERIES
# ─────────────────────────────────────────────────────────────────────────────

async def save_summary(
    summary_id: str,
    chat_id: str,
    content: str,
    covers_up_to: str,
) -> None:
    """
    Persist a newly generated summary to the summaries table.
    Called by the memory injector after it compresses old messages.

    Args:
        summary_id:   UUID4 string for this summary row.
        chat_id:      UUID4 string — which conversation this covers.
        content:      The LLM-generated summary text.
        covers_up_to: message_id of the last message included in this summary.
    Returns: None
    Side effects: Writes one row to summaries.
    """
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO summaries (summary_id, chat_id, content, covers_up_to)
            VALUES (?, ?, ?, ?)
            """,
            (summary_id, chat_id, content, covers_up_to)
        )
        await db.commit()
    logger.debug("Saved summary %s for chat %s", summary_id, chat_id)


async def get_latest_summary(chat_id: str) -> Optional[Summary]:
    """
    Return the most recent summary for a chat, or None if none exist.
    The memory injector uses this to prepend compressed history to the prompt.

    Args:
        chat_id: UUID4 string identifying the conversation.
    Returns:
        Optional[Summary]: the newest summary row, or None.
    """
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT summary_id, chat_id, content, covers_up_to, created_at
            FROM summaries
            WHERE chat_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (chat_id,)
        )
        row = await cursor.fetchone()  # fetchone returns None if no rows

    if row is None:
        return None

    return Summary(
        summary_id=row["summary_id"],
        chat_id=row["chat_id"],
        content=row["content"],
        covers_up_to=row["covers_up_to"],
        created_at=row["created_at"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# METRICS QUERIES
# ─────────────────────────────────────────────────────────────────────────────

async def get_token_stats_for_session(session_id: str) -> dict:
    """Legacy stats query — kept for backwards compatibility."""
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT COUNT(*) as total_messages, COALESCE(SUM(m.token_count), 0) as total_tokens
            FROM messages m
            JOIN chats c ON m.chat_id = c.chat_id
            WHERE c.session_id = ?
            """,
            (session_id,)
        )
        row = await cursor.fetchone()

    total_messages = row["total_messages"] if row else 0
    total_tokens = row["total_tokens"] if row else 0
    avg = total_tokens // total_messages if total_messages > 0 else 0

    return {
        "total_messages": total_messages,
        "total_tokens": total_tokens,
        "avg_tokens_per_message": avg,
    }


async def get_full_metrics_for_session(session_id: str) -> dict:
    """
    Rich analytics query for the dashboard. Returns:
      - total_queries, total_chats, total_tokens
      - tokens_saved, avg_reduction_pct (from rewriter)
      - category_breakdown: {category: count}
      - model_usage: {short_model_name: count}
      - savings_timeline: [{date: "YYYY-MM-DD", saved: N}] per day
    """
    async with get_db() as db:
        # ── Total chat count ──────────────────────────────────────────────────
        cur = await db.execute(
            "SELECT COUNT(*) as n FROM chats WHERE session_id = ?", (session_id,)
        )
        total_chats = (await cur.fetchone())["n"]

        # ── Assistant messages aggregate ──────────────────────────────────────
        cur = await db.execute(
            """
            SELECT
                COUNT(*) as total_queries,
                COALESCE(SUM(m.token_count), 0) as total_tokens,
                COALESCE(SUM(m.original_tokens - m.rewritten_tokens), 0) as tokens_saved,
                COALESCE(AVG(CASE WHEN m.reduction_pct > 0 THEN m.reduction_pct ELSE NULL END), 0) as avg_reduction_pct
            FROM messages m
            JOIN chats c ON m.chat_id = c.chat_id
            WHERE c.session_id = ? AND m.role = 'assistant'
            """,
            (session_id,)
        )
        agg = await cur.fetchone()

        # ── Category breakdown ────────────────────────────────────────────────
        cur = await db.execute(
            """
            SELECT route_category, COUNT(*) as cnt
            FROM messages m
            JOIN chats c ON m.chat_id = c.chat_id
            WHERE c.session_id = ? AND m.role = 'assistant' AND m.route_category IS NOT NULL
            GROUP BY route_category
            ORDER BY cnt DESC
            """,
            (session_id,)
        )
        cat_rows = await cur.fetchall()
        category_breakdown = {}
        for row in cat_rows:
            # compound categories stored as CSV e.g. "dsa,math" — split and count each
            for cat in (row["route_category"] or "").split(","):
                cat = cat.strip()
                if cat:
                    category_breakdown[cat] = category_breakdown.get(cat, 0) + row["cnt"]

        # ── Model usage ───────────────────────────────────────────────────────
        cur = await db.execute(
            """
            SELECT model_used, COUNT(*) as cnt
            FROM messages m
            JOIN chats c ON m.chat_id = c.chat_id
            WHERE c.session_id = ? AND m.role = 'assistant' AND m.model_used IS NOT NULL
            GROUP BY model_used
            ORDER BY cnt DESC
            LIMIT 10
            """,
            (session_id,)
        )
        model_rows = await cur.fetchall()
        model_usage = {}
        for row in model_rows:
            # Shorten the model name for display: "groq/qwen/qwen3-32b" → "qwen3-32b"
            name = (row["model_used"] or "unknown").split("/")[-1]
            model_usage[name] = model_usage.get(name, 0) + row["cnt"]

        # ── Daily savings timeline ────────────────────────────────────────────
        cur = await db.execute(
            """
            SELECT
                DATE(m.created_at) as day,
                COALESCE(SUM(m.original_tokens - m.rewritten_tokens), 0) as saved
            FROM messages m
            JOIN chats c ON m.chat_id = c.chat_id
            WHERE c.session_id = ? AND m.role = 'assistant'
            GROUP BY DATE(m.created_at)
            ORDER BY day ASC
            LIMIT 30
            """,
            (session_id,)
        )
        timeline_rows = await cur.fetchall()
        savings_timeline = [
            {"date": row["day"], "saved": max(0, row["saved"])}
            for row in timeline_rows
        ]

    return {
        "total_queries":      agg["total_queries"] if agg else 0,
        "total_chats":        total_chats,
        "total_tokens":       agg["total_tokens"] if agg else 0,
        "tokens_saved":       max(0, agg["tokens_saved"]) if agg else 0,
        "avg_reduction_pct":  round(agg["avg_reduction_pct"] or 0.0, 1),
        "category_breakdown": category_breakdown,
        "model_usage":        model_usage,
        "savings_timeline":   savings_timeline,
    }
