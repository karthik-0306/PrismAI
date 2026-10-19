"""
backend/routers/history.py

Conversation history endpoints. Every endpoint is scoped to the caller's
session_id: a chat that belongs to another session is reported as 404, so a
leaked or guessed chat_id can't be read, renamed or deleted.

GET    /chats?session_id=…                    sidebar list, most recently active first
GET    /chats/search?session_id=…&q=…         search titles and message content
GET    /chats/{chat_id}/messages?session_id=… full message list, oldest first
PATCH  /chats/{chat_id}?session_id=…          rename  {"title": "..."}
DELETE /chats/{chat_id}?session_id=…          delete chat, messages and summaries
"""

import json
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.database import queries
from backend.utils.session import validate_chat_id, validate_session_id

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatSummary(BaseModel):
    chat_id: str
    title: str
    created_at: str
    updated_at: Optional[str] = None


class MessageItem(BaseModel):
    message_id: str
    role: str
    content: str
    model_used: Optional[str] = None
    route_category: Optional[str] = None
    token_count: int
    created_at: str
    categories_used: List[str] = []
    models_used: List[str] = []
    original_tokens: int = 0
    rewritten_tokens: int = 0
    reduction_pct: float = 0.0


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


def _require_session(session_id: str) -> None:
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id. Must be a UUID4 string.")


async def _require_owned_chat(chat_id: str, session_id: str) -> None:
    _require_session(session_id)
    if not validate_chat_id(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat_id. Must be a UUID4 string.")
    if not await queries.chat_belongs_to(chat_id, session_id):
        raise HTTPException(status_code=404, detail="Chat not found.")


def _to_summary(chat) -> ChatSummary:
    return ChatSummary(chat_id=chat.chat_id, title=chat.title,
                       created_at=chat.created_at, updated_at=chat.updated_at)


def _models_list(msg) -> List[str]:
    if msg.models_used_json:
        try:
            return [m for m in json.loads(msg.models_used_json) if m]
        except (ValueError, TypeError):
            pass
    return [msg.model_used] if msg.model_used else []


@router.get("/chats", response_model=List[ChatSummary], summary="Get all chats for a session")
async def get_chats(session_id: str = Query(...)) -> List[ChatSummary]:
    _require_session(session_id)
    return [_to_summary(c) for c in await queries.get_all_chats_for_session(session_id)]


@router.get("/chats/search", response_model=List[ChatSummary], summary="Search chats by keyword")
async def search_chats_api(q: str = Query(..., min_length=1, max_length=200),
                           session_id: str = Query(...)) -> List[ChatSummary]:
    _require_session(session_id)
    return [_to_summary(c) for c in await queries.search_chats(session_id, q.strip())]


@router.get("/chats/{chat_id}/messages", response_model=List[MessageItem], summary="Get all messages in a chat")
async def get_messages(chat_id: str, session_id: str = Query(...)) -> List[MessageItem]:
    await _require_owned_chat(chat_id, session_id)
    return [
        MessageItem(
            message_id=msg.message_id,
            role=msg.role,
            content=msg.content,
            model_used=msg.model_used,
            route_category=msg.route_category,
            token_count=msg.token_count,
            created_at=msg.created_at,
            categories_used=[c.strip() for c in (msg.route_category or "").split(",") if c.strip()],
            models_used=_models_list(msg),
            original_tokens=msg.original_tokens,
            rewritten_tokens=msg.rewritten_tokens,
            reduction_pct=msg.reduction_pct,
        )
        for msg in await queries.get_chat_messages(chat_id)
    ]


@router.patch("/chats/{chat_id}", response_model=ChatSummary, summary="Rename a chat")
async def rename_chat_api(chat_id: str, body: RenameRequest, session_id: str = Query(...)) -> ChatSummary:
    await _require_owned_chat(chat_id, session_id)
    await queries.rename_chat(chat_id, " ".join(body.title.split()))
    return _to_summary(await queries.get_chat(chat_id))


@router.delete("/chats/{chat_id}", summary="Delete a chat")
async def delete_chat_api(chat_id: str, session_id: str = Query(...)):
    await _require_owned_chat(chat_id, session_id)
    await queries.delete_chat(chat_id)
    return {"status": "success", "message": "Chat deleted successfully"}
