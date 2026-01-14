"""
Video API Schemas
=================
Request/response models for video upload and processing endpoints.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class ProcessingMode(str, Enum):
    """
    Available video processing modes.
    Each mode uses different detection models and prompts.
    """
    TRAFFIC = "traffic"
    FACTORY = "factory"
    KITCHEN = "kitchen"
    GENERAL = "general"


class VideoStatus(str, Enum):
    """
    Video processing status values.
    """
    PENDING = "pending"      # Uploaded, waiting to process
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"   # Processing finished successfully
    FAILED = "failed"        # Processing failed
    SKIPPED = "skipped"      # Already processed (duplicate)


class VideoUploadResponse(BaseModel):
    """
    Response after uploading a video.
    
    Example:
        {
            "video_name": "factory_cam_01.mp4",
            "status": "processing",
            "mode": "factory",
            "message": "Video upload successful, processing started"
        }
    """
    video_name: str = Field(description="Name of the uploaded video file")
    status: VideoStatus = Field(description="Current processing status")
    mode: ProcessingMode = Field(description="Processing mode being used")
    message: str = Field(description="Status message")
    estimated_duration: Optional[float] = Field(default=None, description="Estimated processing time in seconds")


class VideoStatusResponse(BaseModel):
    """
    Detailed status of a video.
    
    Example:
        {
            "video_name": "factory_cam_01.mp4",
            "status": "completed",
            "scene_count": 12,
            "processing_time": 45.2,
            "processed_at": "2024-01-15T10:30:00Z"
        }
    """
    video_name: str = Field(description="Video file name")
    status: VideoStatus = Field(description="Current status")
    mode: Optional[ProcessingMode] = Field(default=None, description="Processing mode used")
    scene_count: Optional[int] = Field(default=None, description="Number of scenes extracted")
    detection_count: Optional[int] = Field(default=None, description="Total objects detected")
    face_count: Optional[int] = Field(default=None, description="Total faces detected")
    duration_seconds: Optional[float] = Field(default=None, description="Video duration")
    processing_time: Optional[float] = Field(default=None, description="Time taken to process (seconds)")
    processed_at: Optional[datetime] = Field(default=None, description="When processing completed")
    error_message: Optional[str] = Field(default=None, description="Error details if failed")


class VideoInfo(BaseModel):
    """
    Basic video information for list responses.
    """
    video_name: str = Field(description="Video file name")
    status: VideoStatus = Field(description="Processing status")
    mode: Optional[ProcessingMode] = Field(default=None)
    scene_count: Optional[int] = Field(default=None)
    file_size_mb: Optional[float] = Field(default=None, description="File size in MB")
    uploaded_at: Optional[datetime] = Field(default=None)


class VideoListResponse(BaseModel):
    """
    List of videos with pagination.
    
    Example:
        {
            "videos": [...],
            "total": 25,
            "page": 1,
            "page_size": 10
        }
    """
    videos: List[VideoInfo] = Field(description="List of videos")
    total: int = Field(description="Total number of videos")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")


class VideoProcessingResult(BaseModel):
    """
    Full result of video processing (internal use).
    Maps to the dict returned by VideoIngestor.process_video()
    """
    status: str
    video_name: str
    duration: Optional[float] = None
    frame_count: Optional[int] = None
    scene_count: Optional[int] = None
    detection_count: Optional[int] = None
    face_count: Optional[int] = None
    processing_time: Optional[float] = None
    message: str
    reason: Optional[str] = None  # For skipped videos
