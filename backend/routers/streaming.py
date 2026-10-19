"""
backend/routers/streaming.py

POST /api/chat/stream — Server-Sent Events (SSE) streaming endpoint.

Each event is one line:  data: <json>\\n\\n   and the stream ends with  data: [DONE]
See backend/pipeline/orchestrator.py for the event types. Every query type —
including compound, multi-part queries — streams through this endpoint.

If the client disconnects (e.g. the user presses "Stop"), Starlette cancels the
generator and the orchestrator saves the partial answer.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import config
from backend.pipeline.orchestrator import Orchestrator
from backend.utils.session import validate_chat_id, validate_session_id

logger = logging.getLogger(__name__)

router = APIRouter()
_orchestrator = Orchestrator()


class ChatRequest(BaseModel):
    """Shared by POST /chat and POST /chat/stream."""
    session_id:       str           = Field(..., description="UUID4 browser session identifier")
    message:          str           = Field(..., min_length=1, max_length=config.MAX_MESSAGE_CHARS)
    chat_id:          Optional[str] = Field(None, description="Existing chat UUID, or null for a new chat")
    model_preference: str           = Field("auto", description="'auto' or a model id from GET /api/models")
    rewriter_enabled: bool          = Field(True)
    regenerate:       bool          = Field(False, description="Replace the chat's last turn instead of appending")


def validate_chat_request(request: ChatRequest) -> dict:
    """Validate and normalise a request into orchestrator kwargs (raises 400)."""
    if not validate_session_id(request.session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id. Must be a UUID4 string.")
    if request.chat_id is not None and not validate_chat_id(request.chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat_id. Must be a UUID4 string.")
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    model = request.model_preference or "auto"
    if model != "auto" and model not in config.ALL_MODELS:
        # Stale value from an old frontend, or an attempt to call an arbitrary model.
        logger.warning("Unknown model_preference %r — using auto routing", model)
        model = "auto"

    return {
        "session_id": request.session_id,
        "chat_id": request.chat_id,
        "message": request.message,
        "model_preference": model,
        "rewriter_enabled": request.rewriter_enabled,
        "regenerate": request.regenerate,
    }


@router.post("/chat/stream", summary="Send a message and stream the response via SSE",
             response_class=StreamingResponse)
async def stream_chat(request: ChatRequest):
    kwargs = validate_chat_request(request)
    logger.info("POST /chat/stream | session=%s | chat=%s | model=%s | len=%d | regen=%s",
                request.session_id, request.chat_id, kwargs["model_preference"],
                len(request.message), request.regenerate)

    async def event_generator():
        try:
            async for event in _orchestrator.stream(**kwargs):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("Streaming pipeline crashed")
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Internal error. Please try again.'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # disable proxy buffering (nginx / Render)
            "Connection": "keep-alive",
        },
    )
