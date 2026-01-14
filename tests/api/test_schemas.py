"""
Schema Validation Tests
=======================
Tests for Pydantic schema validation.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

# Import schemas
from src.api.schemas.common import (
    APIResponse,
    ErrorResponse,
    HealthResponse,
    PaginationParams,
)
from src.api.schemas.video import (
    VideoUploadResponse,
    VideoStatusResponse,
    VideoListResponse,
    VideoInfo,
    VideoStatus,
    ProcessingMode,
)
from src.api.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)


class TestAPIResponseSchema:
    """Tests for APIResponse schema."""
    
    def test_minimal_valid_response(self):
        """APIResponse with minimal required fields."""
        response = APIResponse(message="Success")
        assert response.success is True
        assert response.message == "Success"
        assert response.data is None
    
    def test_full_response(self):
        """APIResponse with all fields."""
        response = APIResponse(
            success=True,
            message="Test message",
            data={"key": "value"},
            request_id="abc123"
        )
        assert response.data == {"key": "value"}
        assert response.request_id == "abc123"
    
    def test_timestamp_auto_generated(self):
        """APIResponse should auto-generate timestamp."""
        response = APIResponse(message="Test")
        assert response.timestamp is not None
        assert isinstance(response.timestamp, datetime)


class TestErrorResponseSchema:
    """Tests for ErrorResponse schema."""
    
    def test_error_response_creation(self):
        """ErrorResponse should require error and error_code."""
        response = ErrorResponse(
            error="Something went wrong",
            error_code="TEST_ERROR"
        )
        assert response.success is False
        assert response.error == "Something went wrong"
        assert response.error_code == "TEST_ERROR"
    
    def test_error_response_with_details(self):
        """ErrorResponse can include details dict."""
        response = ErrorResponse(
            error="Bad request",
            error_code="BAD_REQUEST",
            details={"field": "query", "issue": "too long"}
        )
        assert response.details["field"] == "query"


class TestHealthResponseSchema:
    """Tests for HealthResponse schema."""
    
    def test_health_response_creation(self):
        """HealthResponse with required fields."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            environment="dev",
            databases={"chromadb": "connected", "neo4j": "connected"}
        )
        assert response.status == "healthy"
        assert response.databases["chromadb"] == "connected"


class TestVideoStatusEnum:
    """Tests for VideoStatus enum."""
    
    def test_all_status_values(self):
        """VideoStatus should have all expected values."""
        assert VideoStatus.PENDING == "pending"
        assert VideoStatus.PROCESSING == "processing"
        assert VideoStatus.COMPLETED == "completed"
        assert VideoStatus.FAILED == "failed"
        assert VideoStatus.SKIPPED == "skipped"


class TestProcessingModeEnum:
    """Tests for ProcessingMode enum."""
    
    def test_all_mode_values(self):
        """ProcessingMode should have all expected values."""
        assert ProcessingMode.TRAFFIC == "traffic"
        assert ProcessingMode.FACTORY == "factory"
        assert ProcessingMode.KITCHEN == "kitchen"
        assert ProcessingMode.GENERAL == "general"


class TestVideoUploadResponseSchema:
    """Tests for VideoUploadResponse schema."""
    
    def test_upload_response_creation(self):
        """VideoUploadResponse with required fields."""
        response = VideoUploadResponse(
            video_name="test.mp4",
            status=VideoStatus.PROCESSING,
            mode=ProcessingMode.FACTORY,
            message="Processing started"
        )
        assert response.video_name == "test.mp4"
        assert response.status == VideoStatus.PROCESSING
        assert response.mode == ProcessingMode.FACTORY


class TestSearchRequestSchema:
    """Tests for SearchRequest schema validation."""
    
    def test_valid_search_request(self):
        """SearchRequest with valid data."""
        request = SearchRequest(
            query="safety violations",
            mode=ProcessingMode.FACTORY,
            top_k=5
        )
        assert request.query == "safety violations"
        assert request.mode == ProcessingMode.FACTORY
        assert request.top_k == 5
    
    def test_query_min_length(self):
        """SearchRequest query should have min length 1."""
        with pytest.raises(ValidationError):
            SearchRequest(query="")
    
    def test_query_max_length(self):
        """SearchRequest query should have max length 500."""
        long_query = "a" * 501
        with pytest.raises(ValidationError):
            SearchRequest(query=long_query)
    
    def test_top_k_min_value(self):
        """SearchRequest top_k should have min value 1."""
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=0)
    
    def test_top_k_max_value(self):
        """SearchRequest top_k should have max value 20."""
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=21)
    
    def test_default_values(self):
        """SearchRequest should have correct defaults."""
        request = SearchRequest(query="test")
        assert request.mode is None
        assert request.top_k == 5
    
    def test_optional_mode(self):
        """SearchRequest mode should be optional."""
        request = SearchRequest(query="test", mode=None)
        assert request.mode is None


class TestSearchResultSchema:
    """Tests for SearchResult schema."""
    
    def test_search_result_creation(self):
        """SearchResult with all required fields."""
        result = SearchResult(
            video="test.mp4",
            time="0.0s - 5.0s",
            start_time=0.0,
            end_time=5.0,
            description="Test description",
            mode="factory",
            distance=0.25
        )
        assert result.video == "test.mp4"
        assert result.distance == 0.25
        assert result.yolo_tags == []  # Default empty list
        assert result.persons == []  # Default empty list


class TestSearchResponseSchema:
    """Tests for SearchResponse schema."""
    
    def test_search_response_creation(self):
        """SearchResponse with all fields."""
        response = SearchResponse(
            query="test query",
            result_count=2,
            results=[
                SearchResult(
                    video="v1.mp4",
                    time="0-5s",
                    start_time=0,
                    end_time=5,
                    description="desc1",
                    mode="factory",
                    distance=0.1
                ),
                SearchResult(
                    video="v2.mp4",
                    time="5-10s",
                    start_time=5,
                    end_time=10,
                    description="desc2",
                    mode="factory",
                    distance=0.2
                )
            ],
            summary="Test summary"
        )
        assert response.query == "test query"
        assert response.result_count == 2
        assert len(response.results) == 2


class TestPaginationParamsSchema:
    """Tests for PaginationParams schema."""
    
    def test_default_values(self):
        """PaginationParams should have correct defaults."""
        params = PaginationParams()
        assert params.page == 1
        assert params.page_size == 10
    
    def test_page_min_value(self):
        """PaginationParams page should have min value 1."""
        with pytest.raises(ValidationError):
            PaginationParams(page=0)
    
    def test_page_size_min_value(self):
        """PaginationParams page_size should have min value 1."""
        with pytest.raises(ValidationError):
            PaginationParams(page_size=0)
    
    def test_page_size_max_value(self):
        """PaginationParams page_size should have max value 100."""
        with pytest.raises(ValidationError):
            PaginationParams(page_size=101)
    
    def test_offset_calculation(self):
        """PaginationParams offset property should calculate correctly."""
        params = PaginationParams(page=3, page_size=10)
        assert params.offset == 20  # (3-1) * 10


class TestVideoListResponseSchema:
    """Tests for VideoListResponse schema."""
    
    def test_video_list_response(self):
        """VideoListResponse with video items."""
        response = VideoListResponse(
            videos=[
                VideoInfo(
                    video_name="test1.mp4",
                    status=VideoStatus.COMPLETED
                ),
                VideoInfo(
                    video_name="test2.mp4",
                    status=VideoStatus.PROCESSING
                )
            ],
            total=100,
            page=1,
            page_size=10
        )
        assert len(response.videos) == 2
        assert response.total == 100
