"""
backend/routers/metrics.py

GET /api/metrics?session_id=... — Returns rich analytics data for the dashboard.
GET /api/model-status          — Returns live health status of each LLM provider.
"""

import logging
import asyncio
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Optional

from backend.database import queries
from backend.utils.session import validate_session_id

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    total_queries:        int
    total_chats:          int
    total_tokens:         int
    tokens_saved:         int
    avg_reduction_pct:    float
    category_breakdown:   Dict[str, int]   # {"dsa": 12, "math": 5, ...}
    model_usage:          Dict[str, int]   # {"gemini-3.5-flash": 10, ...}
    savings_timeline:     List[Dict]       # [{date, saved}, ...]


class ModelStatusResponse(BaseModel):
    models: List[Dict]   # [{name, status, latency_ms}]


# ─────────────────────────────────────────────────────────────────────────────
# METRICS ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/metrics", response_model=MetricsResponse,
            summary="Get analytics metrics for a session")
async def get_metrics(
    session_id: str = Query(..., description="UUID4 browser session identifier")
) -> MetricsResponse:
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail=f"Invalid session_id.")

    logger.info("GET /metrics | session=%s", session_id)
    data = await queries.get_full_metrics_for_session(session_id)
    return MetricsResponse(**data)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL STATUS ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/model-status", response_model=ModelStatusResponse,
            summary="Get live health status of each LLM provider")
async def get_model_status() -> ModelStatusResponse:
    """
    Pings each provider with a minimal completion and measures latency.
    Returns green/yellow/red status based on response time or failure.
    """
    import time
    import litellm

    PROVIDERS_TO_CHECK = [
        ("Groq Llama-3.1-8b",   "groq/llama-3.1-8b-instant"),
        ("Groq Qwen3-32b",       "groq/qwen/qwen3-32b"),
        ("Gemini 3.5 Flash",     "gemini/gemini-3.5-flash"),
    ]

    async def ping(display_name: str, model: str) -> dict:
        try:
            t0 = time.monotonic()
            await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0,
            )
            ms = int((time.monotonic() - t0) * 1000)
            status = "green" if ms < 3000 else "yellow"
            return {"name": display_name, "model": model, "status": status, "latency_ms": ms}
        except Exception as e:
            logger.warning("Model ping failed for %s: %s", model, str(e)[:80])
            return {"name": display_name, "model": model, "status": "red", "latency_ms": -1}

    results = await asyncio.gather(*[ping(n, m) for n, m in PROVIDERS_TO_CHECK])
    return ModelStatusResponse(models=list(results))
