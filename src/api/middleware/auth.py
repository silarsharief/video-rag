"""
API Key Authentication Middleware
=================================
Validates API keys on protected endpoints.

How it works:
1. Client sends request with header: X-API-Key: your-key-here
2. This middleware checks if the key is in our allowed list
3. If valid -> request proceeds
4. If invalid -> 401 Unauthorized response
"""

import logging
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config.settings import API_KEY_HEADER, API_KEYS

# Setup logger
logger = logging.getLogger(__name__)

# Define the API key header scheme
# This tells FastAPI to look for the key in request headers
api_key_header = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,  # Don't auto-raise error, we handle it ourselves
    description="API key for authentication. Get this from your admin."
)


async def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """
    Extract API key from request header.
    Used as a dependency in route handlers.
    
    Usage in routes:
        @router.get("/protected")
        async def protected_route(api_key: str = Depends(get_api_key)):
            ...
    """
    return api_key


async def api_key_auth(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Validate API key and raise error if invalid.
    Use this as a dependency for protected routes.
    
    Usage in routes:
        @router.get("/protected")
        async def protected_route(api_key: str = Depends(api_key_auth)):
            # This only runs if API key is valid
            ...
    
    Raises:
        HTTPException 401: If no API key provided
        HTTPException 403: If API key is invalid
    """
    # Check if API key was provided
    if api_key is None:
        logger.warning("Request without API key attempted")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "API key required",
                "error_code": "AUTH_KEY_MISSING",
                "hint": f"Include header: {API_KEY_HEADER}: your-api-key"
            }
        )
    
    # Check if API keys are configured
    if not API_KEYS:
        logger.error("No API keys configured on server!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Server misconfigured",
                "error_code": "AUTH_NOT_CONFIGURED"
            }
        )
    
    # Validate the provided key
    if api_key not in API_KEYS:
        # Log the attempt (mask the key for security)
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
        logger.warning(f"Invalid API key attempted: {masked_key}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Invalid API key",
                "error_code": "AUTH_KEY_INVALID"
            }
        )
    
    # Key is valid
    logger.debug(f"Valid API key authenticated")
    return api_key


def is_public_route(path: str) -> bool:
    """
    Check if a route should be publicly accessible (no auth needed).
    
    Public routes:
    - /api/v1/health - Health check (for load balancers)
    - /docs - Swagger documentation
    - /redoc - ReDoc documentation
    - /openapi.json - OpenAPI schema
    """
    public_paths = [
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/"
    ]
    return any(path.endswith(p) for p in public_paths)
