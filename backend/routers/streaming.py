"""
backend/routers/streaming.py

POST /api/chat/stream — Server-Sent Events (SSE) streaming endpoint.

Emits a stream of newline-delimited SSE events:
  data: {"type": "token",    "content": "Hello "}
  data: {"type": "token",    "content": "world!"}
  data: {"type": "metadata", "chat_id": "...", "categories_used": [...], ...}
  data: [DONE]

Special event:
  data: {"type": "fallback", "chat_id": "..."} — compound query, frontend
        should immediately call POST /api/chat for the non-streaming response.

The frontend reads this stream using the Fetch API with a ReadableStream reader.
"""

import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from backend.pipeline.orchestrator import Orchestrator
from backend.utils.session import validate_session_id

logger = logging.getLogger(__name__)

router = APIRouter()
_orchestrator = Orchestrator()


class StreamRequest(BaseModel):
    """Same fields as ChatRequest — kept in sync."""
    session_id:        str  = Field(..., description="UUID4 browser session identifier")
    message:           str  = Field(..., min_length=1)
    chat_id:           Optional[str]  = Field(None)
    model_preference:  str  = Field("auto")
    rewriter_enabled:  bool = Field(True)


@router.post(
    "/chat/stream",
    summary="Send a message and stream back the LLM response via SSE",
    response_class=StreamingResponse,
)
async def stream_chat(request: StreamRequest):
    """
    SSE streaming endpoint.

    Returns a text/event-stream response where each line is:
      data: <json>\n\n

    The client should use fetch() + ReadableStream to consume this.
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    if not validate_session_id(request.session_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid session_id: '{request.session_id}'."
        )
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    logger.info(
        "POST /chat/stream | session=%s | chat=%s | model_pref=%s | msg_len=%d",
        request.session_id, request.chat_id, request.model_preference, len(request.message)
    )

    async def event_generator():
        try:
            async for event in _orchestrator.stream(
                session_id=request.session_id,
                chat_id=request.chat_id,
                message=request.message,
                model_preference=request.model_preference,
                rewriter_enabled=request.rewriter_enabled,
            ):
                yield f"data: {json.dumps(event)}\n\n"

            # SSE terminator
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Streaming pipeline error: %s", e, exc_info=True)
            error_event = {"type": "error", "detail": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if deployed
        },
    )
