"""
backend/main.py

The main entry point for the PrismAI FastAPI backend.

Responsibilities:
  1. Configure CORS for the frontend (Vite dev server + FRONTEND_URL in production).
  2. Initialise the database schema on startup.
  3. Register all API routers under the "/api" prefix.
  4. Return clean JSON for unexpected errors instead of HTML 500 pages.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import config
from backend.database.connection import initialize_database
from backend.llm import embeddings
from backend.routers.chat import router as chat_router
from backend.routers.history import router as history_router
from backend.routers.metrics import router as metrics_router
from backend.routers.streaming import router as streaming_router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# LiteLLM and httpx are extremely chatty at INFO.
for noisy in ("LiteLLM", "litellm", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PrismAI backend…")
    await initialize_database()

    missing = [k for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY") if not os.getenv(k, "").strip()]
    if missing:
        logger.warning("Missing API keys: %s — dependent models/features will be skipped.", ", ".join(missing))
    logger.info("Models: smart=%s coder=%s fast=%s gemini=%s",
                config.SMART, config.CODER, config.FAST, config.GEMINI)
    # Background thread — the server starts accepting requests immediately.
    embeddings.warm_up_in_background()
    yield
    logger.info("Shutting down PrismAI backend…")
    # Let in-flight background work (summaries, partial-answer saves) finish.
    from backend.pipeline.orchestrator import _background
    if _background:
        await asyncio.wait(list(_background), timeout=10)


app = FastAPI(
    title="PrismAI Orchestrator API",
    description="Multi-Agent LLM Orchestration Platform Backend",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# FRONTEND_URL may be a comma-separated list (production + custom domain).
# PRISMAI_CORS_ORIGIN_REGEX optionally allows e.g. Vercel preview deployments:
#   https://prism-ai-.*\.vercel\.app
origins = ["http://localhost:5173", "http://127.0.0.1:5173", *config.FRONTEND_URLS]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=os.getenv("PRISMAI_CORS_ORIGIN_REGEX") or None,
    allow_credentials=False,   # the app uses no cookies; session id travels in the request
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again."})


app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(streaming_router, prefix="/api", tags=["Streaming"])
app.include_router(history_router, prefix="/api", tags=["History"])
app.include_router(metrics_router, prefix="/api", tags=["Metrics"])


@app.get("/health", tags=["Health"])
@app.get("/api/health", tags=["Health"], include_in_schema=False)
async def health_check():
    """Liveness check — no LLM calls, no DB writes."""
    return {"status": "healthy", "service": "PrismAI"}
