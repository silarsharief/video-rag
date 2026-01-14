"""
API Routes (Endpoints)
======================
Each file handles a group of related endpoints.

Contains:
- health.py : Health check and status endpoints
- video.py  : Video upload and processing endpoints
- search.py : Search and query endpoints
"""

from src.api.routes.health import router as health_router
from src.api.routes.video import router as video_router
from src.api.routes.search import router as search_router

__all__ = [
    "health_router",
    "video_router", 
    "search_router",
]
