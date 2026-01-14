"""
Search Endpoint Tests
=====================
Tests for /api/v1/search/* endpoints.
"""

import pytest
from fastapi.testclient import TestClient


class TestSearchEndpoint:
    """Test suite for main search endpoint."""
    
    def test_search_requires_auth(self, client: TestClient):
        """Search should require authentication."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test query"}
        )
        assert response.status_code == 401
    
    def test_search_basic_query(
        self, client: TestClient, auth_headers: dict
    ):
        """Search with basic query should work."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "show me safety violations"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "result_count" in data
        assert "summary" in data
    
    def test_search_response_structure(
        self, client: TestClient, auth_headers: dict
    ):
        """Search response should have correct structure."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "test"}
        )
        
        data = response.json()
        # Required fields
        assert "query" in data
        assert "results" in data
        assert "result_count" in data
        assert "summary" in data
        
        # Results should be a list
        assert isinstance(data["results"], list)
        assert isinstance(data["result_count"], int)
    
    def test_search_with_mode_filter(
        self, client: TestClient, auth_headers: dict
    ):
        """Search with mode filter should apply filter."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "workers", "mode": "factory"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode_filter"] == "factory"
    
    def test_search_with_top_k(
        self, client: TestClient, auth_headers: dict
    ):
        """Search with top_k should limit results."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "test", "top_k": 3}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 3
    
    def test_search_empty_query_fails(
        self, client: TestClient, auth_headers: dict
    ):
        """Search with empty query should fail validation."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": ""}
        )
        
        # Should fail with 422 Unprocessable Entity
        assert response.status_code == 422
    
    def test_search_query_too_long_fails(
        self, client: TestClient, auth_headers: dict
    ):
        """Search with very long query should fail."""
        long_query = "a" * 1000  # Exceeds max_length=500
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": long_query}
        )
        
        assert response.status_code == 422
    
    def test_search_invalid_mode_fails(
        self, client: TestClient, auth_headers: dict
    ):
        """Search with invalid mode should fail validation."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "test", "mode": "invalid_mode"}
        )
        
        # Should fail with 422 Unprocessable Entity
        assert response.status_code == 422
    
    def test_search_top_k_out_of_range(
        self, client: TestClient, auth_headers: dict
    ):
        """Search with top_k out of range should fail."""
        # top_k below minimum
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "test", "top_k": 0}
        )
        assert response.status_code == 422
        
        # top_k above maximum
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "test", "top_k": 100}
        )
        assert response.status_code == 422
    
    def test_search_result_item_structure(
        self, client: TestClient, auth_headers: dict
    ):
        """Search results should have correct item structure."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "safety"}
        )
        
        data = response.json()
        if data["results"]:  # Only check if we have results
            result = data["results"][0]
            # Check expected fields in result item
            assert "video" in result
            assert "time" in result
            assert "start_time" in result
            assert "end_time" in result
            assert "description" in result
            assert "mode" in result
            assert "distance" in result
    
    def test_search_includes_request_id(
        self, client: TestClient, auth_headers: dict
    ):
        """Search response should include request ID in headers."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "test"}
        )
        
        assert "X-Request-ID" in response.headers


class TestSearchStats:
    """Test suite for search stats endpoint."""
    
    def test_stats_requires_auth(self, client: TestClient):
        """Stats should require authentication."""
        response = client.get("/api/v1/search/stats")
        assert response.status_code == 401
    
    def test_stats_returns_success(
        self, client: TestClient, auth_headers: dict
    ):
        """Stats should return successful response."""
        response = client.get("/api/v1/search/stats", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
    
    def test_stats_response_structure(
        self, client: TestClient, auth_headers: dict
    ):
        """Stats should return expected data structure."""
        response = client.get("/api/v1/search/stats", headers=auth_headers)
        data = response.json()
        
        if data["success"]:
            stats = data["data"]
            assert "total_scenes" in stats
            assert "total_videos" in stats
            assert "scenes_by_mode" in stats
    
    def test_stats_scenes_by_mode_structure(
        self, client: TestClient, auth_headers: dict
    ):
        """Stats scenes_by_mode should be a dict."""
        response = client.get("/api/v1/search/stats", headers=auth_headers)
        data = response.json()
        
        if data["success"] and data["data"]:
            assert isinstance(data["data"]["scenes_by_mode"], dict)


class TestSearchValidation:
    """Test input validation for search endpoint."""
    
    def test_search_requires_json_body(
        self, client: TestClient, auth_headers: dict
    ):
        """Search without JSON body should fail."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers
        )
        assert response.status_code == 422
    
    def test_search_requires_query_field(
        self, client: TestClient, auth_headers: dict
    ):
        """Search without query field should fail."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"mode": "factory"}
        )
        assert response.status_code == 422
    
    def test_search_accepts_null_mode(
        self, client: TestClient, auth_headers: dict
    ):
        """Search with null mode should work (search all modes)."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "test", "mode": None}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode_filter"] is None
    
    def test_search_default_top_k(
        self, client: TestClient, auth_headers: dict
    ):
        """Search without top_k should use default value."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "test"}
        )
        
        # Default top_k is 5
        assert response.status_code == 200
