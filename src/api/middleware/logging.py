"""
Request Logging Middleware
==========================
Logs all API requests with unique request IDs for tracing.

Features:
- Generates unique request ID for each request
- Logs request method, path, and timing
- Adds request ID to response headers (X-Request-ID)
- Helps trace issues across logs

Why request IDs matter:
- When user reports an issue, they can give you the request ID
- You can search logs for that ID and see exactly what happened
- Essential for production debugging
"""

import uuid
import time
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from config.settings import API_LOG_REQUESTS, API_LOG_RESPONSES

# Setup logger with custom format
logger = logging.getLogger("api.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all requests and adds request IDs.
    
    How it works:
    1. Request comes in -> generate request ID
    2. Store request ID in request.state (accessible in route handlers)
    3. Log request details
    4. Call the route handler
    5. Log response details
    6. Add request ID to response headers
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process each request through the middleware.
        
        Args:
            request: The incoming HTTP request
            call_next: Function to call the next middleware/route handler
            
        Returns:
            Response with added headers
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]  # Short ID for readability
        
        # Store request ID in request state (accessible in route handlers)
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        
        # Extract request info
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        
        # Log incoming request
        if API_LOG_REQUESTS:
            logger.info(
                f"[{request_id}] --> {method} {path} | client={client_ip}"
            )
        
        # Process the request (call route handler)
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            if API_LOG_RESPONSES:
                logger.info(
                    f"[{request_id}] <-- {method} {path} | "
                    f"status={response.status_code} | "
                    f"time={process_time:.3f}s"
                )
            
            # Add request ID to response headers (for client reference)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}s"
            
            return response
            
        except Exception as e:
            # Log errors with request ID
            process_time = time.time() - start_time
            logger.error(
                f"[{request_id}] !!! {method} {path} | "
                f"error={type(e).__name__}: {str(e)} | "
                f"time={process_time:.3f}s"
            )
            raise


def get_request_id(request: Request) -> str:
    """
    Get request ID from request state.
    Use in route handlers to include request ID in responses.
    
    Usage:
        @router.get("/example")
        async def example(request: Request):
            request_id = get_request_id(request)
            return {"request_id": request_id, ...}
    """
    return getattr(request.state, "request_id", "unknown")
