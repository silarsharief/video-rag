"""
Search Routes
=============
Endpoints for searching analyzed video content.

All endpoints require API key authentication.
"""

import logging
import asyncio
import time
from fastapi import APIRouter, Depends, HTTPException, Request, status

from config.settings import SEARCH_TIMEOUT
from src.api.schemas.common import APIResponse
from src.api.schemas.search import SearchRequest, SearchResponse, SearchResult
from src.api.middleware.auth import api_key_auth
from src.api.middleware.logging import get_request_id
from src.core.retrieval import ForensicSearch

# Create router for search endpoints
router = APIRouter(
    prefix="/search",
    tags=["Search"],
    dependencies=[Depends(api_key_auth)],  # All routes require auth
)

logger = logging.getLogger(__name__)


@router.post(
    "",
    response_model=SearchResponse,
    summary="Search Videos",
    description="Search analyzed video content using natural language."
)
async def search_videos(
    request: Request,
    search_request: SearchRequest,
    api_key: str = Depends(api_key_auth)
) -> SearchResponse:
    """
    Search for relevant video scenes using natural language.
    
    How it works:
    1. Your query is enhanced by AI for better matching
    2. Vector search finds similar scenes in ChromaDB
    3. Results are filtered and ranked by relevance
    4. AI generates a summary of findings
    
    Timeout: Search will terminate after SEARCH_TIMEOUT seconds.
    
    Example:
        POST /api/v1/search
        {
            "query": "show me safety violations",
            "mode": "factory",
            "top_k": 5
        }
    """
    request_id = get_request_id(request)
    query = search_request.query
    mode = search_request.mode.value if search_request.mode else None
    
    logger.info(f"[{request_id}] Search: query='{query}', mode={mode}, timeout={SEARCH_TIMEOUT}s")
    
    # Track search timing
    start_time = time.time()
    
    try:
        # Run search with timeout
        def run_search():
            """Synchronous search function to run in executor."""
            search_engine = ForensicSearch()
            return search_engine.search(query, mode_filter=mode)
        
        # Execute search with timeout using asyncio
        try:
            loop = asyncio.get_event_loop()
            summary, evidence = await asyncio.wait_for(
                loop.run_in_executor(None, run_search),
                timeout=SEARCH_TIMEOUT
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[{request_id}] Search timeout after {elapsed:.2f}s (limit: {SEARCH_TIMEOUT}s)")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "error": f"Search timed out after {SEARCH_TIMEOUT} seconds",
                    "error_code": "SEARCH_TIMEOUT",
                    "request_id": request_id
                }
            )
        
        # Convert evidence to SearchResult objects
        results = []
        for ev in evidence[:search_request.top_k]:
            results.append(SearchResult(
                video=ev.get('video', 'Unknown'),
                time=ev.get('time', 'N/A'),
                start_time=ev.get('start_time', 0),
                end_time=ev.get('end_time', 0),
                description=ev.get('description', ''),
                mode=ev.get('mode', 'unknown'),
                distance=ev.get('distance', 1.0),
                yolo_tags=ev.get('yolo_tags', []),
                persons=ev.get('persons', [])
            ))
        
        # Log timing
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Search complete: {len(results)} results in {elapsed:.2f}s")
        
        return SearchResponse(
            query=query,
            result_count=len(results),
            results=results,
            summary=summary,
            mode_filter=mode
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like timeout)
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Search error after {elapsed:.2f}s: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Search failed: {str(e)}",
                "error_code": "SEARCH_ERROR",
                "request_id": request_id
            }
        )


@router.get(
    "/stats",
    response_model=APIResponse,
    summary="Get Search Stats",
    description="Get statistics about indexed video content."
)
async def get_search_stats(
    request: Request,
    api_key: str = Depends(api_key_auth)
) -> APIResponse:
    """
    Get statistics about what's been indexed.
    
    Returns:
    - Total scenes indexed
    - Videos processed
    - Scenes by mode
    """
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] Stats requested")
    
    try:
        from src.core.database import ForensicDB
        db = ForensicDB()
        
        # Get all items from ChromaDB (be careful with large datasets)
        # For production, use count() or aggregation if available
        all_results = db.vector_col.get(include=["metadatas"])
        
        total_scenes = len(all_results['ids']) if all_results['ids'] else 0
        
        # Count by mode
        mode_counts = {}
        videos = set()
        if all_results['metadatas']:
            for meta in all_results['metadatas']:
                mode = meta.get('mode', 'unknown')
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                videos.add(meta.get('video_name', 'unknown'))
        
        db.close()
        
        return APIResponse(
            success=True,
            message="Statistics retrieved successfully",
            data={
                "total_scenes": total_scenes,
                "total_videos": len(videos),
                "scenes_by_mode": mode_counts,
                "videos": list(videos)
            },
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Stats error: {e}")
        return APIResponse(
            success=False,
            message=f"Failed to get stats: {str(e)}",
            request_id=request_id
        )
