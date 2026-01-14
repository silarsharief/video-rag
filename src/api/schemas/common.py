"""
Common API Schemas
==================
Shared response models used across all endpoints.
"""

from typing import Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """
    Standard API response wrapper.
    All successful responses follow this structure.
    
    Example:
        {
            "success": true,
            "message": "Video uploaded successfully",
            "data": {...},
            "request_id": "abc123"
        }
    """
    success: bool = Field(default=True, description="Whether the request succeeded")
    message: str = Field(description="Human-readable status message")
    data: Optional[Any] = Field(default=None, description="Response payload")
    request_id: Optional[str] = Field(default=None, description="Unique request identifier for tracing")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ErrorResponse(BaseModel):
    """
    Standard error response.
    All error responses follow this structure.
    
    Example:
        {
            "success": false,
            "error": "Invalid API key",
            "error_code": "AUTH_INVALID_KEY",
            "details": {"provided_key": "xxx..."}
        }
    """
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Human-readable error message")
    error_code: str = Field(description="Machine-readable error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error context")
    request_id: Optional[str] = Field(default=None, description="Request ID for support")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """
    Health check response.
    Shows status of all system components.
    
    Example:
        {
            "status": "healthy",
            "version": "1.0.0",
            "databases": {
                "chromadb": "connected",
                "neo4j": "connected"
            }
        }
    """
    status: str = Field(description="Overall health: healthy, degraded, unhealthy")
    version: str = Field(description="API version")
    environment: str = Field(description="Current environment: dev, staging, production")
    databases: Dict[str, str] = Field(description="Status of each database connection")
    uptime_seconds: Optional[float] = Field(default=None, description="Server uptime in seconds")


class PaginationParams(BaseModel):
    """
    Pagination parameters for list endpoints.
    """
    page: int = Field(default=1, ge=1, description="Page number (starts at 1)")
    page_size: int = Field(default=10, ge=1, le=100, description="Items per page (max 100)")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size
