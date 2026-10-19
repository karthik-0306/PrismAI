"""
backend/database/models.py

Python dataclass definitions that mirror the SQLite schema exactly.
These are used throughout the codebase as typed return values from queries.py.
Rule: if you change a field here, change the SQL in connection.py and queries.py too.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Chat:
    """One conversation thread. session_id ties chats to one browser."""
    chat_id: str
    session_id: str
    title: str
    created_at: str
    updated_at: Optional[str] = None   # last activity — drives sidebar ordering


@dataclass
class Message:
    """One turn (user query OR assistant response) in a chat."""
    message_id: str
    chat_id: str
    role: str                       # 'user' or 'assistant'
    content: str
    model_used: Optional[str]       # LiteLLM model string, None for user messages
    route_category: Optional[str]   # e.g. 'dsa', 'dsa,math', None for user messages
    token_count: int
    is_summarized: bool
    created_at: str
    # JSON list of every model that contributed (compound answers use several).
    models_used_json: Optional[str] = None
    original_tokens: int = 0
    rewritten_tokens: int = 0
    reduction_pct: float = 0.0


@dataclass
class Summary:
    """A compressed summary of a batch of older messages."""
    summary_id: str
    chat_id: str
    content: str
    covers_up_to: str   # message_id of the last message included in this summary
    created_at: str
