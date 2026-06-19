"""
backend/routers/chat.py

POST /chat — The primary endpoint. Receives the user's message and returns
the LLM's response along with metadata (model used, route, etc.).

Request body (JSON):
    session_id       str   — UUID4, the browser's persistent session identifier
    message          str   — the user's raw input text
    chat_id          str?  — UUID4 of existing chat, or null to start a new one
    model_preference str?  — "auto" (default) or a specific model string
    rewriter_enabled bool? — Phase 2 flag; accepted but ignored in Phase 1

Response body (JSON):
    response         str   — the assistant's reply text
    model_used       str   — LiteLLM model string that produced the response
    chat_id          str   — UUID4 (same as input if existing, new one if null was sent)
    message_id       str   — UUID4 of the assistant's message row in the DB
    route_category   str   — which ROUTE_MAP bucket was used
    eval_score       float?— Phase 6: quality score from evaluator (null in Phase 1)
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from backend.pipeline.orchestrator import Orchestrator     # the pipeline coordinator
from backend.utils.session import validate_session_id      # input validation

logger = logging.getLogger(__name__)

# ── APIRouter — registered in main.py with prefix "/api" ─────────────────────
router = APIRouter()

# ── Module-level orchestrator singleton — stateless, safe to share ─────────────
_orchestrator = Orchestrator()


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Pydantic model for validating the POST /chat request body.
    FastAPI will return 422 Unprocessable Entity automatically if validation fails.
    """
    session_id: str = Field(..., description="UUID4 browser session identifier")
    message: str = Field(..., min_length=1, description="User's input text")
    chat_id: Optional[str] = Field(None, description="Existing chat UUID, or null for new chat")
    model_preference: str = Field("auto", description="'auto' or a specific LiteLLM model string")
    rewriter_enabled: bool = Field(True, description="Phase 2 flag; ignored in Phase 1")


class ChatResponse(BaseModel):
    """
    Pydantic model for the POST /chat response body.
    Includes all metadata the frontend needs to render the message bubble.
    """
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
    eval_score: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="Send a message and get an LLM response")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint. Validates input, runs the full pipeline, returns result.

    Steps performed:
      1. Validate session_id format (reject bad UUIDs early).
      2. Validate message is non-empty (already handled by Pydantic min_length=1).
      3. Delegate all pipeline logic to the Orchestrator.
      4. Return the result as a ChatResponse.

    Raises:
        422: if request body is malformed (Pydantic handles this automatically).
        400: if session_id is not a valid UUID4.
        500: if the LLM pipeline fails entirely (all models exhausted).
    """
    # ── Validate session_id ──────────────────────────────────────────────────
    if not validate_session_id(request.session_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid session_id: '{request.session_id}'. Must be a UUID4 string."
        )

    # ── Validate message is not just whitespace ──────────────────────────────
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty or whitespace only."
        )

    logger.info(
        "POST /chat | session=%s | chat=%s | model_pref=%s | msg_len=%d",
        request.session_id, request.chat_id, request.model_preference, len(request.message)
    )

    # ── Run the pipeline ─────────────────────────────────────────────────────
    try:
        result = await _orchestrator.run(
            session_id=request.session_id,
            chat_id=request.chat_id,
            message=request.message,
            model_preference=request.model_preference,
            rewriter_enabled=request.rewriter_enabled,
        )
    except Exception as e:
        # Catch-all for LLMError and any unexpected failures
        logger.error("Pipeline failure: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"AI pipeline failed: {str(e)}"
        )

    return ChatResponse(
        response=result.response,
        model_used=result.model_used,
        chat_id=result.chat_id,
        message_id=result.message_id,
        route_category=result.route_category,
        categories_used=result.categories_used,
        models_used=result.models_used,
        original_tokens=result.original_tokens,
        rewritten_tokens=result.rewritten_tokens,
        reduction_pct=result.reduction_pct,
        eval_score=result.eval_score,
    )
