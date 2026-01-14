"""
Server Protection Middleware
============================
Protects the server from overload, abuse, and crashes.

Features:
- Request timeout: Terminate slow requests
- Rate limiting: Prevent API abuse
- Concurrent task limiting: Control background task count
- Circuit breaker: Stop calling failing services
"""

import time
import asyncio
import logging
from typing import Dict, Callable, Optional
from collections import defaultdict
from functools import wraps
from datetime import datetime, timedelta

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import HTTPException, status

from config.settings import (
    REQUEST_TIMEOUT,
    API_RATE_LIMIT_ENABLED,
    API_RATE_LIMIT_REQUESTS,
    API_RATE_LIMIT_WINDOW,
    MAX_CONCURRENT_PROCESSING,
)

logger = logging.getLogger(__name__)


# ============================================================================
# RATE LIMITER (In-Memory)
# ============================================================================
class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.
    
    For production with multiple workers, use Redis-based rate limiting.
    """
    
    def __init__(self):
        # Dict: api_key -> list of request timestamps
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            key: Identifier (API key or IP)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            tuple: (is_allowed, remaining_requests)
        """
        async with self._lock:
            now = time.time()
            window_start = now - window_seconds
            
            # Remove old timestamps outside the window
            self._requests[key] = [
                ts for ts in self._requests[key] 
                if ts > window_start
            ]
            
            current_count = len(self._requests[key])
            
            if current_count >= limit:
                # Rate limit exceeded
                return False, 0
            
            # Add current request
            self._requests[key].append(now)
            remaining = limit - current_count - 1
            
            return True, remaining
    
    def get_reset_time(self, key: str, window_seconds: int) -> int:
        """Get seconds until rate limit resets."""
        if key not in self._requests or not self._requests[key]:
            return 0
        oldest = min(self._requests[key])
        reset_at = oldest + window_seconds
        return max(0, int(reset_at - time.time()))


# Global rate limiter instance
_rate_limiter = RateLimiter()


async def check_rate_limit(api_key: str) -> tuple[bool, int, int]:
    """
    Check if request is within rate limit.
    
    Returns:
        tuple: (is_allowed, remaining, reset_seconds)
    """
    if not API_RATE_LIMIT_ENABLED:
        return True, -1, 0
    
    allowed, remaining = await _rate_limiter.is_allowed(
        api_key, 
        API_RATE_LIMIT_REQUESTS, 
        API_RATE_LIMIT_WINDOW
    )
    reset_time = _rate_limiter.get_reset_time(api_key, API_RATE_LIMIT_WINDOW)
    
    return allowed, remaining, reset_time


# ============================================================================
# CONCURRENT TASK TRACKER
# ============================================================================
class ConcurrentTaskTracker:
    """
    Track and limit concurrent background tasks.
    Prevents server overload from too many video processing jobs.
    """
    
    def __init__(self, max_tasks: int = MAX_CONCURRENT_PROCESSING):
        self.max_tasks = max_tasks
        self._active_tasks: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
    
    async def can_start_task(self, task_id: str) -> bool:
        """
        Check if we can start a new task.
        
        Returns:
            bool: True if under limit, False if at capacity
        """
        async with self._lock:
            # Clean up stale tasks (older than 1 hour - probably stuck)
            now = datetime.now()
            stale_threshold = now - timedelta(hours=1)
            self._active_tasks = {
                tid: started 
                for tid, started in self._active_tasks.items()
                if started > stale_threshold
            }
            
            if len(self._active_tasks) >= self.max_tasks:
                return False
            
            self._active_tasks[task_id] = now
            return True
    
    async def complete_task(self, task_id: str):
        """Mark a task as completed."""
        async with self._lock:
            self._active_tasks.pop(task_id, None)
    
    @property
    def active_count(self) -> int:
        """Get current number of active tasks."""
        return len(self._active_tasks)
    
    @property
    def available_slots(self) -> int:
        """Get number of available task slots."""
        return max(0, self.max_tasks - len(self._active_tasks))


# Global task tracker instance
task_tracker = ConcurrentTaskTracker()


# ============================================================================
# TIMEOUT DECORATOR
# ============================================================================
def with_timeout(timeout_seconds: float):
    """
    Decorator to add timeout to async functions.
    
    Usage:
        @with_timeout(30)
        async def slow_operation():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout after {timeout_seconds}s in {func.__name__}")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail={
                        "error": f"Request timed out after {timeout_seconds} seconds",
                        "error_code": "REQUEST_TIMEOUT"
                    }
                )
        return wrapper
    return decorator


# ============================================================================
# TIMEOUT MIDDLEWARE
# ============================================================================
class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce request timeouts.
    
    Terminates requests that take longer than the configured timeout.
    """
    
    def __init__(self, app, timeout: float = REQUEST_TIMEOUT):
        super().__init__(app)
        self.timeout = timeout
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply timeout to request processing."""
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {request.method} {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={
                    "detail": {
                        "error": f"Request timed out after {self.timeout} seconds",
                        "error_code": "REQUEST_TIMEOUT"
                    }
                }
            )


# ============================================================================
# RATE LIMIT MIDDLEWARE
# ============================================================================
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce rate limiting.
    
    Adds rate limit headers to responses:
    - X-RateLimit-Limit: Maximum requests allowed
    - X-RateLimit-Remaining: Requests remaining
    - X-RateLimit-Reset: Seconds until limit resets
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limit and add headers."""
        if not API_RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        # Get API key from header (or use IP as fallback)
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            # Use IP for unauthenticated requests
            api_key = f"ip:{request.client.host}" if request.client else "unknown"
        
        # Check rate limit
        allowed, remaining, reset_time = await check_rate_limit(api_key)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for: {api_key[:20]}...")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": {
                        "error": "Rate limit exceeded",
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "retry_after": reset_time
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(API_RATE_LIMIT_REQUESTS),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(reset_time)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(API_RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response
