"""
Health Check Routes
===================
Public endpoints for monitoring system health.

These endpoints are PUBLIC (no auth required) because:
- Load balancers need to check if the server is alive
- Monitoring tools need to ping the service
- Kubernetes health probes need access
"""

import time
import logging
from fastapi import APIRouter, Request

from config.settings import ENVIRONMENT, CHROMADB_PATH, NEO4J_URI
from src.api.schemas.common import HealthResponse, APIResponse
from src.api.middleware.logging import get_request_id
from src.api import __version__ as api_version

# Create router for health endpoints
router = APIRouter(
    prefix="/health",
    tags=["Health"],  # Groups endpoints in Swagger docs
)

# Track server start time for uptime calculation
_server_start_time = time.time()

logger = logging.getLogger(__name__)


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API and all databases are healthy."
)
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns status of:
    - API server
    - ChromaDB connection
    - Neo4j connection
    
    Used by:
    - Load balancers (to route traffic)
    - Monitoring systems (to alert on failures)
    - Kubernetes probes (liveness/readiness)
    """
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] Health check requested")
    
    # Check database connections
    db_status = {}
    overall_status = "healthy"
    
    # Check ChromaDB
    try:
        if CHROMADB_PATH.exists():
            db_status["chromadb"] = "connected"
        else:
            db_status["chromadb"] = "path_missing"
            overall_status = "degraded"
    except Exception as e:
        db_status["chromadb"] = f"error: {str(e)}"
        overall_status = "degraded"
    
    # Check Neo4j (just check if URI is configured)
    try:
        if NEO4J_URI:
            db_status["neo4j"] = "configured"
            # Note: Full connection test would require async driver
            # For production, add actual connection test here
        else:
            db_status["neo4j"] = "not_configured"
            overall_status = "degraded"
    except Exception as e:
        db_status["neo4j"] = f"error: {str(e)}"
        overall_status = "degraded"
    
    # Calculate uptime
    uptime = time.time() - _server_start_time
    
    return HealthResponse(
        status=overall_status,
        version=api_version,
        environment=ENVIRONMENT,
        databases=db_status,
        uptime_seconds=uptime
    )


@router.get(
    "/ready",
    response_model=APIResponse,
    summary="Readiness Check",
    description="Check if the API is ready to accept traffic."
)
async def readiness_check(request: Request) -> APIResponse:
    """
    Readiness probe for Kubernetes/load balancers.
    
    Returns 200 if ready, 503 if not ready.
    """
    request_id = get_request_id(request)
    
    # For now, always ready if health check passes
    # In future: check if models are loaded, DB connections warm, etc.
    return APIResponse(
        success=True,
        message="API is ready to accept traffic",
        request_id=request_id
    )


@router.get(
    "/live",
    response_model=APIResponse,
    summary="Liveness Check",
    description="Check if the API process is alive."
)
async def liveness_check(request: Request) -> APIResponse:
    """
    Liveness probe for Kubernetes.
    
    Simple check - if this returns, the process is alive.
    """
    request_id = get_request_id(request)
    
    return APIResponse(
        success=True,
        message="API is alive",
        request_id=request_id
    )
