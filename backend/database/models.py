"""
backend/database/models.py

Python dataclass definitions that mirror the SQLite schema exactly.
These are used throughout the codebase as typed return values from queries.py.
Rule: if you change a field here, change the SQL in connection.py and queries.py too.
"""

from dataclasses import dataclass  # lightweight typed containers, no ORM overhead
from typing import Optional         # for nullable fields that may be None from the DB


@dataclass
class Chat:
    """
    Represents one conversation thread.
    Created when the user sends the first message in a new chat.
    session_id ties multiple chats to the same browser session.
    title is auto-generated from the first 6 words of the first user message.
    """
    chat_id: str           # UUID4 string — primary key
    session_id: str        # UUID4 string — the browser's persistent session
    title: str             # short label shown in the sidebar
    created_at: str        # ISO-8601 timestamp string from SQLite's datetime('now')


@dataclass
class Message:
    """
    Represents one turn (user query OR assistant response) in a chat.
    Both the user message and the LLM reply are stored as separate rows,
    linked by the same chat_id.
    """
    message_id: str             # UUID4 string — primary key
    chat_id: str                # foreign key → chats.chat_id
    role: str                   # 'user' or 'assistant'
    content: str                # raw text of the message
    model_used: Optional[str]   # LiteLLM model string, None for user messages
    route_category: Optional[str]  # e.g. 'dsa', 'dsa,math', None for user messages
    token_count: int            # estimated token count of this message's content
    is_summarized: bool         # True once this message has been rolled into a summary
    created_at: str             # ISO-8601 timestamp string
    # JSON-serialized list of all models that contributed to this response.
    # e.g. '["groq/openai/gpt-oss-120b","groq/qwen/qwen3-32b"]'
    # None for user messages and rows that predate this column (schema migration v2).
    models_used_json: Optional[str] = None
    original_tokens: int = 0
    rewritten_tokens: int = 0
    reduction_pct: float = 0.0


@dataclass
class Summary:
    """
    Represents a compressed summary of a batch of older messages.
    Stored so that long conversations don't exceed the model's context window.
    The summarizer writes here; the memory injector reads from here.
    """
    summary_id: str    # UUID4 string — primary key
    chat_id: str       # foreign key → chats.chat_id
    content: str       # the summary text produced by the LLM
    covers_up_to: str  # message_id of the last message included in this summary
    created_at: str    # ISO-8601 timestamp string
