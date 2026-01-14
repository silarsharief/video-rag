"""
Video Routes
============
Endpoints for video upload, processing, and status checking.

All endpoints require API key authentication.
"""

import os
import logging
import uuid
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, BackgroundTasks, status

from config.settings import VIDEO_STORAGE_DIR, MAX_CONCURRENT_PROCESSING
from config.modes import list_available_modes
from src.api.schemas.common import APIResponse
from src.api.schemas.video import (
    VideoUploadResponse,
    VideoStatusResponse,
    VideoListResponse,
    VideoInfo,
    VideoStatus,
    ProcessingMode,
)
from src.api.middleware.auth import api_key_auth
from src.api.middleware.logging import get_request_id
from src.api.middleware.protection import task_tracker
from src.core.ingest import VideoIngestor
from src.core.database import ForensicDB

# Create router for video endpoints
router = APIRouter(
    prefix="/video",
    tags=["Video"],
    dependencies=[Depends(api_key_auth)],  # All routes require auth
)

logger = logging.getLogger(__name__)


async def process_video_background(video_path: str, mode: str, task_id: str):
    """
    Background task to process video with task tracking.
    Runs asynchronously after upload response is sent.
    
    Args:
        video_path: Full path to the video file
        mode: Processing mode (traffic, factory, kitchen, general)
        task_id: Unique ID for tracking this task
    """
    try:
        logger.info(f"[{task_id}] Background processing started: {video_path} (mode={mode})")
        logger.info(f"[{task_id}] Active tasks: {task_tracker.active_count}/{MAX_CONCURRENT_PROCESSING}")
        
        ingestor = VideoIngestor(video_path, mode=mode)
        result = ingestor.process_video()
        
        logger.info(f"[{task_id}] Background processing completed: {video_path} - {result.get('status')}")
    except Exception as e:
        logger.error(f"[{task_id}] Background processing failed: {video_path} - {e}")
    finally:
        # Always mark task as complete
        await task_tracker.complete_task(task_id)
        logger.info(f"[{task_id}] Task slot released. Active tasks: {task_tracker.active_count}/{MAX_CONCURRENT_PROCESSING}")


@router.post(
    "/upload",
    response_model=VideoUploadResponse,
    summary="Upload Video",
    description="Upload a video file for processing."
)
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file (MP4, MOV)"),
    mode: ProcessingMode = Form(default=ProcessingMode.GENERAL, description="Processing mode"),
    process_now: bool = Form(default=True, description="Start processing immediately"),
    api_key: str = Depends(api_key_auth)
) -> VideoUploadResponse:
    """
    Upload a video file for forensic analysis.
    
    Steps:
    1. Validate file type
    2. Save to storage directory
    3. Start background processing (if process_now=True)
    4. Return upload status
    
    Example:
        curl -X POST /api/v1/video/upload \\
            -H "X-API-Key: your-key" \\
            -F "file=@video.mp4" \\
            -F "mode=factory"
    """
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] Video upload: {file.filename}, mode={mode.value}")
    
    # Validate file extension (primary check - more reliable than content_type)
    allowed_extensions = [".mp4", ".mov", ".avi"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"Invalid file extension: {file_ext}",
                "error_code": "INVALID_EXTENSION",
                "allowed_extensions": allowed_extensions
            }
        )
    
    # Note: content_type check is optional because curl and many clients
    # send "application/octet-stream" by default. Extension check is sufficient.
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo", "application/octet-stream"]
    if file.content_type and file.content_type not in allowed_types:
        logger.warning(f"[{request_id}] Unexpected content type: {file.content_type}, proceeding with extension check")
    
    # Save file to storage directory
    file_path = VIDEO_STORAGE_DIR / file.filename
    try:
        VIDEO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"[{request_id}] File saved: {file_path}")
        
    except Exception as e:
        logger.error(f"[{request_id}] Failed to save file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Failed to save file: {str(e)}",
                "error_code": "FILE_SAVE_ERROR"
            }
        )
    
    # Check if video was already processed (to avoid duplicate processing)
    already_processed = False
    scene_count = 0
    try:
        db = ForensicDB()
        results = db.vector_col.get(
            where={
                "$and": [
                    {"video_name": file.filename},
                    {"mode": mode.value}
                ]
            }
        )
        db.close()
        if results and results.get('ids'):
            already_processed = True
            scene_count = len(results['ids'])
            logger.info(f"[{request_id}] Video already processed, skipping: {file.filename}")
    except Exception as e:
        logger.warning(f"[{request_id}] Could not check if video processed: {e}")
    
    # Determine status based on whether processing starts
    if already_processed:
        current_status = VideoStatus.COMPLETED
        message = f"Video already processed ({scene_count} scenes). Use force_reprocess to re-analyze."
    elif process_now:
        # Check if we can start a new processing task
        task_id = str(uuid.uuid4())[:8]
        can_start = await task_tracker.can_start_task(task_id)
        
        if not can_start:
            logger.warning(f"[{request_id}] Server at capacity: {task_tracker.active_count}/{MAX_CONCURRENT_PROCESSING} tasks running")
            current_status = VideoStatus.PENDING
            message = f"Server at capacity ({MAX_CONCURRENT_PROCESSING} videos processing). Video queued, try again later."
        else:
            # Add background task for processing
            background_tasks.add_task(process_video_background, str(file_path), mode.value, task_id)
            current_status = VideoStatus.PROCESSING
            message = f"Video uploaded successfully, processing started (task {task_id})"
            logger.info(f"[{request_id}] Started task {task_id}. Active: {task_tracker.active_count}/{MAX_CONCURRENT_PROCESSING}")
    else:
        current_status = VideoStatus.PENDING
        message = "Video uploaded successfully, processing not started"
    
    return VideoUploadResponse(
        video_name=file.filename,
        status=current_status,
        mode=mode,
        message=message
    )


@router.get(
    "/{video_name}/status",
    response_model=VideoStatusResponse,
    summary="Get Video Status",
    description="Get the processing status of a video."
)
async def get_video_status(
    request: Request,
    video_name: str,
    api_key: str = Depends(api_key_auth)
) -> VideoStatusResponse:
    """
    Check the processing status of a video.
    
    Returns:
    - Status (pending, processing, completed, failed)
    - Scene count (if completed)
    - Processing time (if completed)
    - Error message (if failed)
    """
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] Status check: {video_name}")
    
    # Check if video file exists
    video_path = VIDEO_STORAGE_DIR / video_name
    if not video_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": f"Video not found: {video_name}",
                "error_code": "VIDEO_NOT_FOUND"
            }
        )
    
    # Query ChromaDB to see if video has been processed
    try:
        db = ForensicDB()
        results = db.vector_col.get(where={"video_name": video_name})
        db.close()
        
        if results and results.get('ids'):
            # Video has been processed
            scene_count = len(results['ids'])
            # Get mode from first result
            mode = results['metadatas'][0].get('mode') if results['metadatas'] else None
            
            return VideoStatusResponse(
                video_name=video_name,
                status=VideoStatus.COMPLETED,
                mode=ProcessingMode(mode) if mode else None,
                scene_count=scene_count
            )
        else:
            # Video exists but not processed
            return VideoStatusResponse(
                video_name=video_name,
                status=VideoStatus.PENDING,
                scene_count=0
            )
            
    except Exception as e:
        logger.error(f"[{request_id}] Status check error: {e}")
        return VideoStatusResponse(
            video_name=video_name,
            status=VideoStatus.PENDING,
            error_message=str(e)
        )


@router.get(
    "/list",
    response_model=VideoListResponse,
    summary="List Videos",
    description="List all uploaded videos with their status."
)
async def list_videos(
    request: Request,
    page: int = 1,
    page_size: int = 10,
    api_key: str = Depends(api_key_auth)
) -> VideoListResponse:
    """
    List all videos in storage with their processing status.
    
    Supports pagination:
    - page: Page number (starts at 1)
    - page_size: Items per page (default 10, max 100)
    """
    request_id = get_request_id(request)
    logger.info(f"[{request_id}] List videos: page={page}, size={page_size}")
    
    # Validate pagination
    page_size = min(page_size, 100)  # Cap at 100
    
    # Get all video files
    VIDEO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    video_files = list(VIDEO_STORAGE_DIR.glob("*.mp4")) + list(VIDEO_STORAGE_DIR.glob("*.mov"))
    total = len(video_files)
    
    # Apply pagination
    start = (page - 1) * page_size
    end = start + page_size
    paginated_files = video_files[start:end]
    
    # Build video info list
    videos = []
    for vf in paginated_files:
        file_size_mb = vf.stat().st_size / (1024 * 1024)
        videos.append(VideoInfo(
            video_name=vf.name,
            status=VideoStatus.PENDING,  # Would need DB query for actual status
            file_size_mb=round(file_size_mb, 2)
        ))
    
    return VideoListResponse(
        videos=videos,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get(
    "/modes",
    response_model=APIResponse,
    summary="List Available Modes",
    description="Get list of available video processing modes."
)
async def get_available_modes(
    request: Request,
    api_key: str = Depends(api_key_auth)
) -> APIResponse:
    """
    Return available processing modes.
    
    Modes:
    - traffic: Vehicle and pedestrian analysis
    - factory: PPE and safety compliance
    - kitchen: Hygiene and food safety
    - general: General scene description
    """
    request_id = get_request_id(request)
    modes = list_available_modes()
    
    return APIResponse(
        success=True,
        message=f"Found {len(modes)} available modes",
        data={"modes": modes, "default": "general"},
        request_id=request_id
    )


@router.get(
    "/capacity",
    response_model=APIResponse,
    summary="Get Processing Capacity",
    description="Check current server processing capacity."
)
async def get_processing_capacity(
    request: Request,
    api_key: str = Depends(api_key_auth)
) -> APIResponse:
    """
    Get current server processing capacity.
    
    Returns:
    - active_tasks: Number of videos currently processing
    - max_tasks: Maximum concurrent processing limit
    - available_slots: How many more videos can be processed
    - can_accept: Whether server can accept new processing requests
    """
    request_id = get_request_id(request)
    
    active = task_tracker.active_count
    available = task_tracker.available_slots
    
    return APIResponse(
        success=True,
        message=f"Server capacity: {active}/{MAX_CONCURRENT_PROCESSING} tasks running",
        data={
            "active_tasks": active,
            "max_tasks": MAX_CONCURRENT_PROCESSING,
            "available_slots": available,
            "can_accept": available > 0
        },
        request_id=request_id
    )
