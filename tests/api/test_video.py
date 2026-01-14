"""
Video Endpoint Tests
====================
Tests for /api/v1/video/* endpoints.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient


class TestVideoUpload:
    """Test suite for video upload endpoint."""
    
    def test_upload_requires_auth(self, client: TestClient, sample_video_file: Path):
        """Upload should require authentication."""
        with open(sample_video_file, "rb") as f:
            response = client.post(
                "/api/v1/video/upload",
                files={"file": ("test.mp4", f, "video/mp4")}
            )
        assert response.status_code == 401
    
    def test_upload_mp4_succeeds(
        self, client: TestClient, auth_headers: dict, sample_video_file: Path
    ):
        """Uploading valid MP4 should succeed."""
        with open(sample_video_file, "rb") as f:
            response = client.post(
                "/api/v1/video/upload",
                headers=auth_headers,
                files={"file": ("test.mp4", f, "video/mp4")},
                data={"mode": "general", "process_now": "false"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["video_name"] == "test.mp4"
        assert data["status"] in ["pending", "processing", "completed"]
    
    def test_upload_mov_succeeds(
        self, client: TestClient, auth_headers: dict, sample_mov_file: Path
    ):
        """Uploading valid MOV should succeed."""
        with open(sample_mov_file, "rb") as f:
            response = client.post(
                "/api/v1/video/upload",
                headers=auth_headers,
                files={"file": ("test.mov", f, "video/quicktime")},
                data={"mode": "factory", "process_now": "false"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["video_name"] == "test.mov"
    
    def test_upload_invalid_extension_fails(
        self, client: TestClient, auth_headers: dict, invalid_file: Path
    ):
        """Uploading file with invalid extension should fail."""
        with open(invalid_file, "rb") as f:
            response = client.post(
                "/api/v1/video/upload",
                headers=auth_headers,
                files={"file": ("test.txt", f, "text/plain")},
                data={"mode": "general"}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "INVALID_EXTENSION"
    
    def test_upload_with_different_modes(
        self, client: TestClient, auth_headers: dict, sample_video_file: Path
    ):
        """Upload should accept different processing modes."""
        modes = ["traffic", "factory", "kitchen", "general"]
        
        for mode in modes:
            with open(sample_video_file, "rb") as f:
                response = client.post(
                    "/api/v1/video/upload",
                    headers=auth_headers,
                    files={"file": (f"test_{mode}.mp4", f, "video/mp4")},
                    data={"mode": mode, "process_now": "false"}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["mode"] == mode
    
    def test_upload_default_mode_is_general(
        self, client: TestClient, auth_headers: dict, sample_video_file: Path
    ):
        """Upload without mode should default to general."""
        with open(sample_video_file, "rb") as f:
            response = client.post(
                "/api/v1/video/upload",
                headers=auth_headers,
                files={"file": ("test_default.mp4", f, "video/mp4")},
                data={"process_now": "false"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "general"
    
    def test_upload_response_includes_request_id(
        self, client: TestClient, auth_headers: dict, sample_video_file: Path
    ):
        """Upload response should include request ID in headers."""
        with open(sample_video_file, "rb") as f:
            response = client.post(
                "/api/v1/video/upload",
                headers=auth_headers,
                files={"file": ("test_reqid.mp4", f, "video/mp4")},
                data={"process_now": "false"}
            )
        
        assert "X-Request-ID" in response.headers
    
    def test_upload_without_file_fails(
        self, client: TestClient, auth_headers: dict
    ):
        """Upload without file should fail with 422."""
        response = client.post(
            "/api/v1/video/upload",
            headers=auth_headers,
            data={"mode": "general"}
        )
        
        assert response.status_code == 422


class TestVideoStatus:
    """Test suite for video status endpoint."""
    
    def test_status_requires_auth(self, client: TestClient):
        """Status check should require authentication."""
        response = client.get("/api/v1/video/test_video.mp4/status")
        assert response.status_code == 401
    
    def test_status_nonexistent_video_returns_404(
        self, client: TestClient, auth_headers: dict
    ):
        """Status of non-existent video should return 404."""
        response = client.get(
            "/api/v1/video/nonexistent_video_12345.mp4/status",
            headers=auth_headers
        )
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error_code"] == "VIDEO_NOT_FOUND"
    
    def test_status_response_structure(
        self, client: TestClient, auth_headers: dict, sample_video_file: Path
    ):
        """Status response should have correct structure."""
        # First upload a video
        with open(sample_video_file, "rb") as f:
            client.post(
                "/api/v1/video/upload",
                headers=auth_headers,
                files={"file": ("status_test.mp4", f, "video/mp4")},
                data={"process_now": "false"}
            )
        
        # Then check status
        response = client.get(
            "/api/v1/video/status_test.mp4/status",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "video_name" in data
        assert "status" in data


class TestVideoList:
    """Test suite for video list endpoint."""
    
    def test_list_requires_auth(self, client: TestClient):
        """List should require authentication."""
        response = client.get("/api/v1/video/list")
        assert response.status_code == 401
    
    def test_list_returns_correct_structure(
        self, client: TestClient, auth_headers: dict
    ):
        """List should return paginated response."""
        response = client.get("/api/v1/video/list", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "videos" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert isinstance(data["videos"], list)
    
    def test_list_pagination_defaults(
        self, client: TestClient, auth_headers: dict
    ):
        """List should have default pagination values."""
        response = client.get("/api/v1/video/list", headers=auth_headers)
        data = response.json()
        
        assert data["page"] == 1
        assert data["page_size"] == 10
    
    def test_list_pagination_params(
        self, client: TestClient, auth_headers: dict
    ):
        """List should accept pagination parameters."""
        response = client.get(
            "/api/v1/video/list?page=2&page_size=5",
            headers=auth_headers
        )
        data = response.json()
        
        assert data["page"] == 2
        assert data["page_size"] == 5
    
    def test_list_page_size_capped_at_100(
        self, client: TestClient, auth_headers: dict
    ):
        """List page_size should be capped at 100."""
        response = client.get(
            "/api/v1/video/list?page_size=500",
            headers=auth_headers
        )
        data = response.json()
        
        assert data["page_size"] <= 100


class TestVideoModes:
    """Test suite for video modes endpoint."""
    
    def test_modes_requires_auth(self, client: TestClient):
        """Modes endpoint should require authentication."""
        response = client.get("/api/v1/video/modes")
        assert response.status_code == 401
    
    def test_modes_returns_available_modes(
        self, client: TestClient, auth_headers: dict
    ):
        """Modes should return list of available modes."""
        response = client.get("/api/v1/video/modes", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "modes" in data["data"]
        
        modes = data["data"]["modes"]
        assert isinstance(modes, list)
        assert "general" in modes
        assert "factory" in modes
        assert "traffic" in modes
        assert "kitchen" in modes
    
    def test_modes_includes_default(
        self, client: TestClient, auth_headers: dict
    ):
        """Modes should indicate default mode."""
        response = client.get("/api/v1/video/modes", headers=auth_headers)
        data = response.json()
        
        assert "default" in data["data"]
        assert data["data"]["default"] == "general"
