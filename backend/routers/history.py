"""
backend/routers/history.py

Two read-only endpoints for retrieving conversation history.

GET /chats?session_id=...
    Returns the sidebar list — all chats for a session, newest first.
    Each item includes chat_id, title, created_at.

GET /chats/{chat_id}/messages
    Returns the full message list for one chat, oldest first.
    Used when the user clicks a chat in the sidebar to load it.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from backend.database import queries                   # DB read functions
from backend.utils.session import validate_session_id  # input validation

logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ChatSummary(BaseModel):
    """One chat entry as returned in the sidebar list."""
    chat_id: str
    title: str
    created_at: str


class MessageItem(BaseModel):
    """One message as returned in the full chat view."""
    message_id: str
    role: str               # 'user' or 'assistant'
    content: str
    model_used: Optional[str] = None
    route_category: Optional[str] = None
    token_count: int
    created_at: str


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/chats", response_model=List[ChatSummary], summary="Get all chats for a session")
async def get_chats(
    session_id: str = Query(..., description="UUID4 browser session identifier")
) -> List[ChatSummary]:
    """
    Return all chat threads belonging to this session, newest first.
    Used to populate the sidebar on load and after refresh.

    Args:
        session_id: UUID4 string passed as a query parameter.
    Returns:
        List[ChatSummary]: list of chat objects with id, title, created_at.
    Raises:
        400: if session_id is not a valid UUID4.
    """
    if not validate_session_id(session_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid session_id: '{session_id}'. Must be a UUID4 string."
        )

    logger.info("GET /chats | session=%s", session_id)

    chats = await queries.get_all_chats_for_session(session_id)

    return [
        ChatSummary(
            chat_id=chat.chat_id,
            title=chat.title,
            created_at=chat.created_at,
        )
        for chat in chats
    ]


@router.get(
    "/chats/{chat_id}/messages",
    response_model=List[MessageItem],
    summary="Get all messages in a chat"
)
async def get_messages(chat_id: str) -> List[MessageItem]:
    """
    Return all messages for a given chat, oldest first.
    Called when the user clicks a chat in the sidebar.

    Args:
        chat_id: UUID4 string from the URL path.
    Returns:
        List[MessageItem]: full message list for the chat.
    Raises:
        400: if chat_id is not a valid UUID4.
        404: if no messages are found (chat doesn't exist or is empty).
    """
    # Reuse validate_session_id since both use UUID4 format
    from backend.utils.session import validate_chat_id
    if not validate_chat_id(chat_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chat_id: '{chat_id}'. Must be a UUID4 string."
        )

    logger.info("GET /chats/%s/messages", chat_id)

    messages = await queries.get_chat_messages(chat_id)

    return [
        MessageItem(
            message_id=msg.message_id,
            role=msg.role,
            content=msg.content,
            model_used=msg.model_used,
            route_category=msg.route_category,
            token_count=msg.token_count,
            created_at=msg.created_at,
        )
        for msg in messages
    ]
