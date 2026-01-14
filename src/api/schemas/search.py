"""
Search API Schemas
==================
Request/response models for search and query endpoints.
"""

from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field

from src.api.schemas.video import ProcessingMode


class SearchRequest(BaseModel):
    """
    Search request body.
    
    Example:
        {
            "query": "show me safety violations",
            "mode": "factory",
            "top_k": 5
        }
    """
    query: str = Field(
        min_length=1,
        max_length=500,
        description="Natural language search query"
    )
    mode: Optional[ProcessingMode] = Field(
        default=None,
        description="Filter by mode (null = search all modes)"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return"
    )


class SearchResult(BaseModel):
    """
    A single search result (evidence).
    
    Example:
        {
            "video": "factory_01.mp4",
            "time": "12.5s - 18.0s",
            "description": "Worker without hardhat near machinery",
            "mode": "factory",
            "distance": 0.23,
            "yolo_tags": ["person", "machinery"],
            "persons": ["PID_12345"]
        }
    """
    video: str = Field(description="Source video name")
    time: str = Field(description="Time range of the scene")
    start_time: float = Field(description="Scene start time in seconds")
    end_time: float = Field(description="Scene end time in seconds")
    description: str = Field(description="AI-generated scene description")
    mode: str = Field(description="Processing mode of this scene")
    distance: float = Field(description="Similarity distance (lower = more relevant)")
    yolo_tags: List[str] = Field(default=[], description="Detected objects")
    persons: List[str] = Field(default=[], description="Detected person IDs")


class SearchResponse(BaseModel):
    """
    Search response with results and AI summary.
    
    Example:
        {
            "query": "safety violations",
            "enhanced_query": "factory worker safety violations PPE",
            "result_count": 3,
            "results": [...],
            "summary": "Found 3 scenes with safety violations..."
        }
    """
    query: str = Field(description="Original user query")
    enhanced_query: Optional[str] = Field(default=None, description="AI-enhanced query used for search")
    result_count: int = Field(description="Number of results returned")
    results: List[SearchResult] = Field(description="List of matching scenes")
    summary: str = Field(description="AI-generated summary of findings")
    mode_filter: Optional[str] = Field(default=None, description="Mode filter applied")


class AvailableModesResponse(BaseModel):
    """
    List of available processing modes.
    """
    modes: List[str] = Field(description="Available mode names")
    default_mode: str = Field(default="general", description="Default mode if not specified")
