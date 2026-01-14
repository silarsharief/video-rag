"""
Pydantic Schemas for API Request/Response Validation
=====================================================
These models define the structure of data going in and out of the API.

Benefits:
- Automatic validation (bad requests are rejected)
- Auto-generated API documentation (Swagger/OpenAPI)
- Type hints for IDE autocomplete
"""

from src.api.schemas.common import (
    APIResponse,
    ErrorResponse,
    HealthResponse,
)
from src.api.schemas.video import (
    VideoUploadResponse,
    VideoStatusResponse,
    VideoListResponse,
)
from src.api.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)

__all__ = [
    # Common
    "APIResponse",
    "ErrorResponse", 
    "HealthResponse",
    # Video
    "VideoUploadResponse",
    "VideoStatusResponse",
    "VideoListResponse",
    # Search
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
]
