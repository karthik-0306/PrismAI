"""
backend/database/connection.py

Manages the aiosqlite database connection and schema initialisation.

Responsibilities:
  - Provide a reusable async context manager (get_db) for opening/closing connections.
  - Define and create all three tables on first startup via initialize_database().

Why aiosqlite?
  SQLite's built-in sqlite3 module is synchronous — calling it inside an async
  FastAPI route would block the entire event loop. aiosqlite wraps sqlite3 with
  asyncio coroutines so every DB operation yields control back to the event loop
  while waiting for disk I/O.

The DB file lives at prismai.db in the project root (next to main.py).
"""

import logging                    # structured logging instead of print()
import aiosqlite                  # async wrapper around sqlite3
from contextlib import asynccontextmanager  # for the async with get_db() pattern
from pathlib import Path          # cross-platform path construction

# ── Logger for this module ───────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Database file path ───────────────────────────────────────────────────────
# __file__ is backend/database/connection.py
# .parent gives backend/database/, .parent.parent gives backend/, .parent gives project root
DB_PATH = Path(__file__).parent.parent.parent / "prismai.db"


@asynccontextmanager
async def get_db():
    """
    Async context manager that opens an aiosqlite connection and closes it on exit.

    Usage:
        async with get_db() as db:
            await db.execute(...)

    Always enable WAL (Write-Ahead Logging) for better concurrent read/write
    performance — multiple readers can proceed while a write is in progress.

    Yields:
        aiosqlite.Connection: an open database connection
    Side effects:
        Opens and closes an aiosqlite connection to DB_PATH.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        # WAL mode allows reads to proceed concurrently with a single writer
        await db.execute("PRAGMA journal_mode=WAL")
        # Return foreign key enforcement (SQLite disables it by default)
        await db.execute("PRAGMA foreign_keys=ON")
        db.row_factory = aiosqlite.Row  # rows behave like dicts (row["column_name"])
        yield db


async def initialize_database() -> None:
    """
    Creates all three tables if they do not already exist.
    Called once at application startup via FastAPI's lifespan event.
    Safe to call on subsequent restarts — IF NOT EXISTS prevents re-creation.

    Tables created:
        chats     — one row per conversation thread
        messages  — one row per user/assistant turn
        summaries — one row per compressed memory summary

    Returns: None
    Side effects: Creates prismai.db and all tables on disk.
    """
    logger.info("Initialising database at %s", DB_PATH)

    async with get_db() as db:
        # ── chats table ─────────────────────────────────────────────────────
        # One row per conversation. session_id links it to the browser session.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id    TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                title      TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # ── messages table ───────────────────────────────────────────────────
        # One row per turn (user OR assistant). Both sides of every exchange live here.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id     TEXT PRIMARY KEY,
                chat_id        TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                role           TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content        TEXT NOT NULL,
                model_used     TEXT,
                route_category TEXT,
                token_count    INTEGER NOT NULL DEFAULT 0,
                is_summarized  INTEGER NOT NULL DEFAULT 0,
                created_at     TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # ── summaries table ──────────────────────────────────────────────────
        # Each row is an LLM-generated compression of a batch of old messages.
        # covers_up_to lets the memory injector know which messages are already summarised.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id   TEXT PRIMARY KEY,
                chat_id      TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                content      TEXT NOT NULL,
                covers_up_to TEXT NOT NULL,
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # ── indexes for common query patterns ────────────────────────────────
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_chat_id
            ON messages(chat_id, created_at)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_chats_session_id
            ON chats(session_id, created_at)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_summaries_chat_id
            ON summaries(chat_id, created_at)
        """)

        await db.commit()  # persist the CREATE TABLE statements

        # ── Schema migrations — run after every startup, idempotent ─────────
        # SQLite has no ALTER TABLE ... ADD COLUMN IF NOT EXISTS syntax.
        # We catch OperationalError (column already exists) and move on silently.
        # Add new columns here when the schema evolves; never remove old ones.

        try:
            await db.execute(
                "ALTER TABLE messages ADD COLUMN models_used_json TEXT"
            )
            await db.commit()
            logger.info("Migration applied: messages.models_used_json column added.")
        except Exception:
            pass

        try:
            await db.execute("ALTER TABLE messages ADD COLUMN original_tokens INTEGER DEFAULT 0")
            await db.execute("ALTER TABLE messages ADD COLUMN rewritten_tokens INTEGER DEFAULT 0")
            await db.execute("ALTER TABLE messages ADD COLUMN reduction_pct REAL DEFAULT 0.0")
            await db.commit()
            logger.info("Migration applied: token metric columns added.")
        except Exception:
            # Columns already exist
            pass

    logger.info("Database initialised successfully.")
