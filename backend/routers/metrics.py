"""
backend/routers/metrics.py

GET /api/metrics?session_id=... — analytics data for the dashboard.
GET /api/model-status           — health of each configured model.

Model status is read from the LLM client's health registry, which is updated by
every real call the app makes — so it reflects live reality at zero cost.
Models with no recent traffic are pinged with a 1-token request at most once
per PING_TTL_S. (The previous version pinged every provider on every poll,
which itself burned the free-tier rate limits and caused 429s elsewhere.)
"""

import asyncio
import logging
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.database import queries
import litellm

from backend.llm.client import _build_kwargs, _mark_failed, _mark_ok, health_snapshot
from backend.utils.session import validate_session_id

logger = logging.getLogger(__name__)
router = APIRouter()

PING_TTL_S = 300.0
_last_ping: Dict[str, float] = {}
_ping_lock = asyncio.Lock()


class MetricsResponse(BaseModel):
    total_queries:      int
    total_chats:        int
    total_tokens:       int
    tokens_saved:       int
    avg_reduction_pct:  float
    category_breakdown: Dict[str, int]
    model_usage:        Dict[str, int]
    savings_timeline:   List[Dict]


class ModelStatusResponse(BaseModel):
    models: List[Dict]   # [{name, model, status, latency_ms, last_error}]


@router.get("/metrics", response_model=MetricsResponse, summary="Get analytics metrics for a session")
async def get_metrics(session_id: str = Query(...)) -> MetricsResponse:
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id.")
    return MetricsResponse(**await queries.get_full_metrics_for_session(session_id))


async def _ping(model: str) -> None:
    """
    Reachability check. Reasoning models may legitimately return no visible text
    for a tiny budget, so any successful HTTP response counts as healthy (going
    through LLMClient would treat that empty answer as a failure).
    """
    t0 = time.monotonic()
    try:
        await litellm.acompletion(**_build_kwargs(model, [{"role": "user", "content": "Say OK"}],
                                                  temperature=0, max_tokens=16,
                                                  low_reasoning=True, json_mode=False))
        _mark_ok(model, int((time.monotonic() - t0) * 1000))
    except Exception as e:
        _mark_failed(model, e)
        logger.info("Model ping failed for %s: %s", model, str(e)[:120])


@router.get("/model-status", response_model=ModelStatusResponse, summary="Health of each LLM")
async def get_model_status() -> ModelStatusResponse:
    now = time.monotonic()
    async with _ping_lock:
        stale = [s["model"] for s in health_snapshot()
                 if s["status"] == "unknown" and now - _last_ping.get(s["model"], 0) > PING_TTL_S]
        for m in stale:
            _last_ping[m] = now
        if stale:
            await asyncio.gather(*[_ping(m) for m in stale])
    return ModelStatusResponse(models=health_snapshot())
