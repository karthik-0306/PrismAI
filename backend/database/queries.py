"""
backend/database/queries.py

All SQL operations for PrismAI.

Rules enforced in this file:
  1. Every SQL statement uses parameterized queries (?, not f-strings) to prevent injection.
  2. All functions are async — they must be awaited by the caller.
  3. Functions return typed dataclasses (from models.py), not raw aiosqlite.Row objects.
  4. No business logic here — this file only reads/writes the DB.
  5. Messages are always ordered by (created_at, rowid): created_at has one-second
     resolution, so a user message and its fast reply often share a timestamp.
"""

import json
import logging
from typing import List, Optional

from backend.database.connection import get_db
from backend.database.models import Chat, Message, Summary

logger = logging.getLogger(__name__)

_CHAT_COLUMNS = "chat_id, session_id, title, created_at, updated_at"
_MESSAGE_COLUMNS = (
    "message_id, chat_id, role, content, model_used, route_category, token_count, "
    "is_summarized, created_at, models_used_json, original_tokens, rewritten_tokens, reduction_pct"
)


def _row_to_chat(row) -> Chat:
    return Chat(
        chat_id=row["chat_id"],
        session_id=row["session_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"] or row["created_at"],
    )


def _row_to_message(row) -> Message:
    return Message(
        message_id=row["message_id"],
        chat_id=row["chat_id"],
        role=row["role"],
        content=row["content"],
        model_used=row["model_used"],
        route_category=row["route_category"],
        token_count=row["token_count"] or 0,
        is_summarized=bool(row["is_summarized"]),
        created_at=row["created_at"],
        models_used_json=row["models_used_json"],
        original_tokens=row["original_tokens"] or 0,
        rewritten_tokens=row["rewritten_tokens"] or 0,
        reduction_pct=row["reduction_pct"] or 0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHAT QUERIES
# ─────────────────────────────────────────────────────────────────────────────

async def save_chat(chat_id: str, session_id: str, title: str) -> None:
    async with get_db() as db:
        await db.execute(
            "INSERT INTO chats (chat_id, session_id, title, updated_at) VALUES (?, ?, ?, datetime('now'))",
            (chat_id, session_id, title),
        )
        await db.commit()
    logger.debug("Saved chat %s for session %s", chat_id, session_id)


async def get_chat(chat_id: str) -> Optional[Chat]:
    async with get_db() as db:
        cur = await db.execute(f"SELECT {_CHAT_COLUMNS} FROM chats WHERE chat_id = ?", (chat_id,))
        row = await cur.fetchone()
    return _row_to_chat(row) if row else None


async def chat_belongs_to(chat_id: str, session_id: str) -> bool:
    chat = await get_chat(chat_id)
    return chat is not None and chat.session_id == session_id


async def touch_chat(chat_id: str) -> None:
    """Bump updated_at so the chat moves to the top of the sidebar."""
    async with get_db() as db:
        await db.execute("UPDATE chats SET updated_at = datetime('now') WHERE chat_id = ?", (chat_id,))
        await db.commit()


async def rename_chat(chat_id: str, title: str) -> None:
    async with get_db() as db:
        await db.execute("UPDATE chats SET title = ? WHERE chat_id = ?", (title, chat_id))
        await db.commit()


async def get_all_chats_for_session(session_id: str) -> List[Chat]:
    """All chats for a browser session, most recently active first."""
    async with get_db() as db:
        cur = await db.execute(
            f"SELECT {_CHAT_COLUMNS} FROM chats WHERE session_id = ? "
            "ORDER BY COALESCE(updated_at, created_at) DESC, rowid DESC",
            (session_id,),
        )
        rows = await cur.fetchall()
    return [_row_to_chat(r) for r in rows]


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def search_chats(session_id: str, query: str) -> List[Chat]:
    """Chats whose title or any message contains `query` (case-insensitive)."""
    pattern = f"%{_escape_like(query)}%"
    async with get_db() as db:
        cur = await db.execute(
            f"""
            SELECT {_CHAT_COLUMNS} FROM chats c
            WHERE c.session_id = ?
              AND (c.title LIKE ? ESCAPE '\\'
                   OR EXISTS (SELECT 1 FROM messages m
                              WHERE m.chat_id = c.chat_id AND m.content LIKE ? ESCAPE '\\'))
            ORDER BY COALESCE(c.updated_at, c.created_at) DESC, c.rowid DESC
            LIMIT 50
            """,
            (session_id, pattern, pattern),
        )
        rows = await cur.fetchall()
    return [_row_to_chat(r) for r in rows]


async def delete_chat(chat_id: str) -> None:
    """Delete a chat with all its messages and summaries."""
    async with get_db() as db:
        await db.execute("DELETE FROM summaries WHERE chat_id = ?", (chat_id,))
        await db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        await db.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
        await db.commit()
    logger.info("Deleted chat %s", chat_id)


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
    """Insert one message (user or assistant turn)."""
    models_json = json.dumps(models_used) if models_used else None
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO messages
                (message_id, chat_id, role, content, model_used, route_category,
                 token_count, models_used_json, original_tokens, rewritten_tokens, reduction_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, chat_id, role, content, model_used, route_category,
             token_count, models_json, original_tokens, rewritten_tokens, reduction_pct),
        )
        await db.execute("UPDATE chats SET updated_at = datetime('now') WHERE chat_id = ?", (chat_id,))
        await db.commit()
    logger.debug("Saved %s message %s in chat %s", role, message_id, chat_id)


async def get_chat_messages(chat_id: str) -> List[Message]:
    """All messages for a chat, oldest first."""
    async with get_db() as db:
        cur = await db.execute(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages WHERE chat_id = ? ORDER BY created_at ASC, rowid ASC",
            (chat_id,),
        )
        rows = await cur.fetchall()
    return [_row_to_message(r) for r in rows]


async def get_unsummarized_messages(chat_id: str) -> List[Message]:
    """Messages not yet rolled into a summary, oldest first."""
    async with get_db() as db:
        cur = await db.execute(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages "
            "WHERE chat_id = ? AND is_summarized = 0 ORDER BY created_at ASC, rowid ASC",
            (chat_id,),
        )
        rows = await cur.fetchall()
    return [_row_to_message(r) for r in rows]


async def delete_messages(message_ids: List[str]) -> None:
    if not message_ids:
        return
    placeholders = ", ".join("?" for _ in message_ids)
    async with get_db() as db:
        await db.execute(f"DELETE FROM messages WHERE message_id IN ({placeholders})", tuple(message_ids))
        await db.commit()


async def mark_messages_summarized(message_ids: List[str]) -> None:
    if not message_ids:
        return
    placeholders = ", ".join("?" for _ in message_ids)
    async with get_db() as db:
        await db.execute(
            f"UPDATE messages SET is_summarized = 1 WHERE message_id IN ({placeholders})",
            tuple(message_ids),
        )
        await db.commit()
    logger.debug("Marked %d messages as summarized", len(message_ids))


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY QUERIES
# ─────────────────────────────────────────────────────────────────────────────

async def save_summary(summary_id: str, chat_id: str, content: str, covers_up_to: str) -> None:
    async with get_db() as db:
        await db.execute(
            "INSERT INTO summaries (summary_id, chat_id, content, covers_up_to) VALUES (?, ?, ?, ?)",
            (summary_id, chat_id, content, covers_up_to),
        )
        await db.commit()
    logger.debug("Saved summary %s for chat %s", summary_id, chat_id)


async def get_latest_summary(chat_id: str) -> Optional[Summary]:
    async with get_db() as db:
        cur = await db.execute(
            "SELECT summary_id, chat_id, content, covers_up_to, created_at FROM summaries "
            "WHERE chat_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (chat_id,),
        )
        row = await cur.fetchone()
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

def _display_model_name(raw: str) -> str:
    # Legacy rows stored subagent answers as "subagent/dsa (groq/qwen/qwen3.6-27b)" —
    # unwrap to the real model before shortening.
    if "(" in raw and raw.endswith(")"):
        raw = raw[raw.find("(") + 1:-1]
    return raw.split("/")[-1]


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
        cur = await db.execute("SELECT COUNT(*) AS n FROM chats WHERE session_id = ?", (session_id,))
        total_chats = (await cur.fetchone())["n"]

        cur = await db.execute(
            """
            SELECT
                COUNT(*) AS total_queries,
                COALESCE(SUM(m.token_count), 0) AS total_tokens,
                COALESCE(SUM(MAX(m.original_tokens - m.rewritten_tokens, 0)), 0) AS tokens_saved,
                COALESCE(AVG(CASE WHEN m.reduction_pct > 0 THEN m.reduction_pct END), 0) AS avg_reduction_pct
            FROM messages m JOIN chats c ON m.chat_id = c.chat_id
            WHERE c.session_id = ? AND m.role = 'assistant'
            """,
            (session_id,),
        )
        agg = await cur.fetchone()

        cur = await db.execute(
            """
            SELECT route_category, COUNT(*) AS cnt
            FROM messages m JOIN chats c ON m.chat_id = c.chat_id
            WHERE c.session_id = ? AND m.role = 'assistant' AND m.route_category IS NOT NULL
            GROUP BY route_category
            """,
            (session_id,),
        )
        category_breakdown: dict = {}
        for row in await cur.fetchall():
            for cat in (row["route_category"] or "").split(","):
                cat = cat.strip()
                if cat:
                    category_breakdown[cat] = category_breakdown.get(cat, 0) + row["cnt"]

        cur = await db.execute(
            """
            SELECT m.model_used, m.models_used_json
            FROM messages m JOIN chats c ON m.chat_id = c.chat_id
            WHERE c.session_id = ? AND m.role = 'assistant'
            """,
            (session_id,),
        )
        model_usage: dict = {}
        for row in await cur.fetchall():
            models = []
            if row["models_used_json"]:
                try:
                    models = [m for m in json.loads(row["models_used_json"]) if m]
                except (ValueError, TypeError):
                    models = []
            if not models and row["model_used"] and row["model_used"] != "aggregated":
                models = [row["model_used"]]
            for m in models:
                name = _display_model_name(m)
                model_usage[name] = model_usage.get(name, 0) + 1

        cur = await db.execute(
            """
            SELECT DATE(m.created_at) AS day,
                   COALESCE(SUM(MAX(m.original_tokens - m.rewritten_tokens, 0)), 0) AS saved
            FROM messages m JOIN chats c ON m.chat_id = c.chat_id
            WHERE c.session_id = ? AND m.role = 'assistant'
            GROUP BY DATE(m.created_at)
            ORDER BY day DESC
            LIMIT 30
            """,
            (session_id,),
        )
        savings_timeline = [{"date": r["day"], "saved": r["saved"]} for r in await cur.fetchall()]
        savings_timeline.reverse()

    return {
        "total_queries":      agg["total_queries"] if agg else 0,
        "total_chats":        total_chats,
        "total_tokens":       agg["total_tokens"] if agg else 0,
        "tokens_saved":       agg["tokens_saved"] if agg else 0,
        "avg_reduction_pct":  round(agg["avg_reduction_pct"] or 0.0, 1) if agg else 0.0,
        "category_breakdown": dict(sorted(category_breakdown.items(), key=lambda kv: -kv[1])),
        "model_usage":        dict(sorted(model_usage.items(), key=lambda kv: -kv[1])[:10]),
        "savings_timeline":   savings_timeline,
    }
