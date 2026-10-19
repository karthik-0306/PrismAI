"""
backend/routers/chat.py

POST /api/chat — non-streaming variant of /api/chat/stream (same request body).
Runs the identical pipeline and returns the full answer in one JSON response.
Useful for scripts, tests and clients that can't consume SSE.

GET /api/models — the models a user can pick in manual mode.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import config
from backend.pipeline.orchestrator import Orchestrator, PipelineError
from backend.routers.streaming import ChatRequest, validate_chat_request

logger = logging.getLogger(__name__)

router = APIRouter()
_orchestrator = Orchestrator()


class ChatResponse(BaseModel):
    response: str
    model_used: str
    chat_id: str
    message_id: str
    route_category: str
    categories_used: List[str]
    models_used: List[str]
    original_tokens: int
    rewritten_tokens: int
    reduction_pct: float
    title: Optional[str] = None
    sources: List[dict] = []


@router.post("/chat", response_model=ChatResponse, summary="Send a message and get the full response")
async def chat(request: ChatRequest) -> ChatResponse:
    kwargs = validate_chat_request(request)
    logger.info("POST /chat | session=%s | chat=%s | model=%s | len=%d",
                request.session_id, request.chat_id, kwargs["model_preference"], len(request.message))
    try:
        result = await _orchestrator.run(**kwargs)
    except PipelineError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return ChatResponse(**result.__dict__)


@router.get("/models", summary="Models available for manual selection")
async def list_models():
    descriptions = {
        config.SMART:  "Deepest reasoning",
        config.CODER:  "Code & fast answers",
        config.GEMINI: "Google · long context",
        config.FAST:   "Lowest latency",
    }
    return {
        "models": [
            {"value": m, "label": config.model_label(m), "sub": descriptions.get(m, "")}
            for m in config.ALL_MODELS
        ]
    }
