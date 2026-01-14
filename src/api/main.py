"""
FastAPI Application Entry Point
===============================
Main application file that ties everything together.

How to run:
    # Development (with auto-reload)
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
    
    # Production (with multiple workers)
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

Access:
    - API: http://localhost:8000/api/v1/
    - Docs: http://localhost:8000/docs (Swagger UI)
    - ReDoc: http://localhost:8000/redoc (alternative docs)
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import configuration
from config.settings import (
    API_PREFIX,
    API_DEBUG,
    ENVIRONMENT,
    CORS_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS,
    LOG_LEVEL,
)

# Import middleware
from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.protection import TimeoutMiddleware, RateLimitMiddleware

# Import routes
from src.api.routes.health import router as health_router
from src.api.routes.video import router as video_router
from src.api.routes.search import router as search_router

from src.api import __version__

# ============================================================================
# LOGGING SETUP
# ============================================================================
# Configure logging for the API
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api")


# ============================================================================
# LIFESPAN EVENTS (Startup/Shutdown)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown events.
    
    Startup: Log configuration, warm up connections
    Shutdown: Clean up resources, close connections
    """
    # === STARTUP ===
    logger.info("="*60)
    logger.info("🚀 FORENSIC RAG API STARTING")
    logger.info("="*60)
    logger.info(f"Version: {__version__}")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"API Prefix: {API_PREFIX}")
    logger.info(f"Debug Mode: {API_DEBUG}")
    logger.info(f"CORS Origins: {CORS_ORIGINS}")
    logger.info("="*60)
    
    # Yield control to the application
    yield
    
    # === SHUTDOWN ===
    logger.info("="*60)
    logger.info("🛑 FORENSIC RAG API SHUTTING DOWN")
    logger.info("="*60)


# ============================================================================
# CREATE FASTAPI APPLICATION
# ============================================================================
app = FastAPI(
    title="Forensic RAG API",
    description="""
    ## Forensic Video Analysis API
    
    Edge-first forensic video analysis system with:
    - 🎬 Video upload and processing
    - 🔍 Semantic search across video content
    - 🤖 AI-powered scene analysis
    - 📊 Multi-modal detection (traffic, factory, kitchen)
    
    ### Authentication
    All endpoints (except health checks) require an API key.
    Include header: `X-API-Key: your-api-key`
    
    ### Quick Start
    1. Upload a video: `POST /api/v1/video/upload`
    2. Check status: `GET /api/v1/video/{video_name}/status`
    3. Search: `POST /api/v1/search`
    """,
    version=__version__,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_url="/openapi.json",
    lifespan=lifespan,
    debug=API_DEBUG,
)


# ============================================================================
# ADD MIDDLEWARE (Order matters! First added = outermost layer)
# ============================================================================

# 1. CORS Middleware (must be first to handle preflight requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

# 2. Request Timeout Middleware (terminates slow requests)
app.add_middleware(TimeoutMiddleware)

# 3. Rate Limiting Middleware (prevents API abuse)
app.add_middleware(RateLimitMiddleware)

# 4. Request Logging Middleware
app.add_middleware(RequestLoggingMiddleware)


# ============================================================================
# REGISTER ROUTES
# ============================================================================
# Health routes (public, no auth)
app.include_router(health_router, prefix=API_PREFIX)

# Video routes (protected)
app.include_router(video_router, prefix=API_PREFIX)

# Search routes (protected)
app.include_router(search_router, prefix=API_PREFIX)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - redirects to documentation.
    """
    return {
        "name": "Forensic RAG API",
        "version": __version__,
        "docs": "/docs",
        "health": f"{API_PREFIX}/health"
    }


# ============================================================================
# RUN DIRECTLY (for development)
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    from config.settings import API_HOST, API_PORT
    
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,  # Auto-reload on code changes (dev only)
        log_level="info"
    )
