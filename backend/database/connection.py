"""
backend/database/connection.py

Manages the aiosqlite database connection and schema initialisation.

Responsibilities:
  - Provide a reusable async context manager (get_db) for opening/closing connections.
  - Create all tables on first startup and apply idempotent column migrations.

Why aiosqlite?
  SQLite's built-in sqlite3 module is synchronous — calling it inside an async
  FastAPI route would block the entire event loop. aiosqlite wraps sqlite3 with
  asyncio coroutines so every DB operation yields control back to the event loop.

The DB file location comes from backend.config.DB_PATH (env: PRISMAI_DB_PATH).
NOTE: on hosts without a persistent disk (e.g. Render's free tier) this file is
wiped on every restart/redeploy. The app tolerates that (stale chat ids from the
browser are re-created), but history will not survive a restart there.
"""

import logging
from contextlib import asynccontextmanager

import aiosqlite

from backend import config

logger = logging.getLogger(__name__)

DB_PATH = config.DB_PATH


@asynccontextmanager
async def get_db():
    """
    Async context manager that opens an aiosqlite connection and closes it on exit.

    Usage:
        async with get_db() as db:
            await db.execute(...)
    """
    async with aiosqlite.connect(DB_PATH, timeout=15) as db:
        await db.execute("PRAGMA journal_mode=WAL")   # readers don't block the writer
        await db.execute("PRAGMA foreign_keys=ON")    # SQLite disables FKs by default
        await db.execute("PRAGMA busy_timeout=5000")
        db.row_factory = aiosqlite.Row
        yield db


async def _add_column_if_missing(db, table: str, column: str, ddl: str) -> None:
    cur = await db.execute(f"PRAGMA table_info({table})")
    existing = {row["name"] for row in await cur.fetchall()}
    if column not in existing:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        logger.info("Migration applied: %s.%s added", table, column)


async def initialize_database() -> None:
    """
    Create all tables if they do not already exist, then apply column migrations.
    Called once at application startup via FastAPI's lifespan event. Idempotent.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Initialising database at %s", DB_PATH)

    async with get_db() as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id    TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                title      TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

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

        await db.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id   TEXT PRIMARY KEY,
                chat_id      TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                content      TEXT NOT NULL,
                covers_up_to TEXT NOT NULL,
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # ── Column migrations — one check per column so a partially migrated
        #    DB can never skip the remaining columns. Never remove old ones.
        await _add_column_if_missing(db, "messages", "models_used_json", "TEXT")
        await _add_column_if_missing(db, "messages", "original_tokens", "INTEGER DEFAULT 0")
        await _add_column_if_missing(db, "messages", "rewritten_tokens", "INTEGER DEFAULT 0")
        await _add_column_if_missing(db, "messages", "reduction_pct", "REAL DEFAULT 0.0")
        await _add_column_if_missing(db, "chats", "updated_at", "TEXT")
        await db.execute("UPDATE chats SET updated_at = created_at WHERE updated_at IS NULL")

        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id, created_at)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_chats_session_id ON chats(session_id, created_at)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_summaries_chat_id ON summaries(chat_id, created_at)")

        await db.commit()

    logger.info("Database initialised successfully.")
