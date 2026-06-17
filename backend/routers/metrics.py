"""
backend/routers/metrics.py

GET /metrics?session_id=... — Returns token usage statistics for a session.

Phase 1: returns basic aggregate counts from the DB.
Phase 2: will add token savings from the query rewriter (original vs rewritten).
The frontend polls this endpoint every 5 messages to update the MetricsBar.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.database import queries                   # DB aggregation query
from backend.utils.session import validate_session_id  # input validation

logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    """
    Token usage statistics for a session.
    Phase 1 fields (real data):
        total_messages, total_tokens, avg_tokens_per_message
    Phase 2 additions (placeholders until rewriter is built):
        tokens_saved_by_rewriter, avg_reduction_pct, rewrite_count
    """
    total_messages: int
    total_tokens: int
    avg_tokens_per_message: int
    # Phase 2 placeholders — always 0 until the rewriter stores its data
    tokens_saved_by_rewriter: int = 0
    avg_reduction_pct: float = 0.0
    rewrite_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/metrics", response_model=MetricsResponse, summary="Get token usage metrics for a session")
async def get_metrics(
    session_id: str = Query(..., description="UUID4 browser session identifier")
) -> MetricsResponse:
    """
    Return aggregated token usage statistics for all chats in a session.

    Args:
        session_id: UUID4 string passed as a query parameter.
    Returns:
        MetricsResponse: token counts and (Phase 2+) rewriter savings stats.
    Raises:
        400: if session_id is not a valid UUID4.
    """
    if not validate_session_id(session_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid session_id: '{session_id}'. Must be a UUID4 string."
        )

    logger.info("GET /metrics | session=%s", session_id)

    # Fetch real aggregate data from the DB
    stats = await queries.get_token_stats_for_session(session_id)

    return MetricsResponse(
        total_messages=stats["total_messages"],
        total_tokens=stats["total_tokens"],
        avg_tokens_per_message=stats["avg_tokens_per_message"],
        # Rewriter fields stay at 0 until Phase 2
        tokens_saved_by_rewriter=0,
        avg_reduction_pct=0.0,
        rewrite_count=0,
    )
