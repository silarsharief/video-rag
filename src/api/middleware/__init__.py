"""
API Middleware
==============
Middleware runs on every request before/after your route handlers.

Contains:
- auth.py       : API key validation
- logging.py    : Request/response logging with request IDs
- protection.py : Timeouts, rate limiting, task limits
"""

from src.api.middleware.auth import api_key_auth, get_api_key
from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.protection import (
    TimeoutMiddleware,
    RateLimitMiddleware,
    task_tracker,
    with_timeout,
    check_rate_limit,
)

__all__ = [
    "api_key_auth",
    "get_api_key",
    "RequestLoggingMiddleware",
    "TimeoutMiddleware",
    "RateLimitMiddleware",
    "task_tracker",
    "with_timeout",
    "check_rate_limit",
]
