"""
backend/main.py

The main entry point for the PrismAI FastAPI backend.

Responsibilities:
  1. Configures CORS middleware for frontend communication (localhost:5173).
  2. Implements lifespan startup logic to initialize the database schema automatically.
  3. Registers all API routers (chat, history, metrics) under the "/api" prefix.
  4. Configures clean logging format for development.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database.connection import initialize_database
from backend.routers.chat import router as chat_router
from backend.routers.history import router as history_router
from backend.routers.metrics import router as metrics_router
from backend.routers.streaming import router as streaming_router

# ── Configure Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan Handler (Startup / Shutdown events) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown lifecycles of the FastAPI application.
    Automatically initializes the SQLite database schema on boot.
    """
    logger.info("Starting up PrismAI FastAPI backend...")
    
    # Run DB migrations/initialization
    try:
        await initialize_database()
        logger.info("Database initialized successfully during startup lifespan.")
    except Exception as e:
        logger.error("Failed to initialize database during startup: %s", e, exc_info=True)
        raise e
        
    yield
    
    logger.info("Shutting down PrismAI FastAPI backend...")

# ── FastAPI Application Instance ──────────────────────────────────────────────
app = FastAPI(
    title="PrismAI Orchestrator API",
    description="Multi-Agent LLM Orchestration Platform Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware Configuration ──────────────────────────────────────────────
# In development: allow Vite dev server (localhost:5173)
# In production (Render): allow the deployed Vercel frontend URL
# Set FRONTEND_URL env var on Render to your Vercel app URL.
import os

_frontend_url = os.getenv("FRONTEND_URL", "").strip()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Add the production Vercel URL if set (e.g. https://prism-ai.vercel.app)
if _frontend_url:
    origins.append(_frontend_url)
    # Also allow without trailing slash
    origins.append(_frontend_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ──────────────────────────────────────────────────────────
# All API endpoints are prefixed with "/api" to distinguish from frontend routing
app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(history_router, prefix="/api", tags=["History"])
app.include_router(metrics_router, prefix="/api", tags=["Metrics"])
app.include_router(streaming_router, prefix="/api", tags=["Streaming"])

# Root debug health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint to verify backend is responsive."""
    return {"status": "healthy", "service": "PrismAI"}
