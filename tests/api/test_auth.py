"""
Authentication Middleware Tests
===============================
Tests for API key authentication.
"""

import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import (
    VALID_API_KEY,
    INVALID_API_KEY,
    assert_error_response,
    assert_success_response,
)


class TestAPIKeyAuthentication:
    """Test suite for API key authentication."""
    
    def test_request_without_api_key_returns_401(self, client: TestClient):
        """Request without API key should return 401 Unauthorized."""
        response = client.get("/api/v1/video/list")
        assert response.status_code == 401
        data = response.json()
        assert data["detail"]["error_code"] == "AUTH_KEY_MISSING"
    
    def test_request_with_invalid_api_key_returns_403(
        self, client: TestClient, invalid_auth_headers: dict
    ):
        """Request with invalid API key should return 403 Forbidden."""
        response = client.get("/api/v1/video/list", headers=invalid_auth_headers)
        assert response.status_code == 403
        data = response.json()
        assert data["detail"]["error_code"] == "AUTH_KEY_INVALID"
    
    def test_request_with_valid_api_key_succeeds(
        self, client: TestClient, auth_headers: dict
    ):
        """Request with valid API key should succeed."""
        response = client.get("/api/v1/video/list", headers=auth_headers)
        assert response.status_code == 200
    
    def test_request_with_empty_api_key_returns_401(self, client: TestClient):
        """Request with empty API key should return 401."""
        response = client.get(
            "/api/v1/video/list",
            headers={"X-API-Key": ""}
        )
        # Empty string might be treated as missing or invalid
        assert response.status_code in [401, 403]
    
    def test_health_endpoint_does_not_require_auth(self, client: TestClient):
        """Health endpoints should be public (no auth required)."""
        # Main health check
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        # Readiness probe
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
        
        # Liveness probe
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
    
    def test_root_endpoint_does_not_require_auth(self, client: TestClient):
        """Root endpoint should be public."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_docs_endpoint_accessible_without_auth(self, client: TestClient):
        """Swagger docs should be accessible without auth."""
        response = client.get("/docs")
        # Should redirect or return HTML
        assert response.status_code in [200, 307]
    
    def test_api_key_case_sensitive(self, client: TestClient):
        """API key should be case-sensitive."""
        # Use uppercase version of valid key
        response = client.get(
            "/api/v1/video/list",
            headers={"X-API-Key": VALID_API_KEY.upper()}
        )
        # Should fail if key is case-sensitive
        if VALID_API_KEY != VALID_API_KEY.upper():
            assert response.status_code == 403
    
    def test_api_key_with_whitespace_fails(self, client: TestClient):
        """API key with leading/trailing whitespace should fail."""
        response = client.get(
            "/api/v1/video/list",
            headers={"X-API-Key": f" {VALID_API_KEY} "}
        )
        assert response.status_code == 403
    
    def test_multiple_valid_keys_work(self, client: TestClient):
        """All configured API keys should work."""
        # Test first key
        response = client.get(
            "/api/v1/video/list",
            headers={"X-API-Key": "test-api-key-12345"}
        )
        assert response.status_code == 200
        
        # Test second key
        response = client.get(
            "/api/v1/video/list",
            headers={"X-API-Key": "another-test-key"}
        )
        assert response.status_code == 200


class TestAuthErrorMessages:
    """Test that auth errors return helpful messages."""
    
    def test_missing_key_error_includes_hint(self, client: TestClient):
        """Missing API key error should include hint about header."""
        response = client.get("/api/v1/video/list")
        data = response.json()
        assert "hint" in data["detail"]
        assert "X-API-Key" in data["detail"]["hint"]
    
    def test_invalid_key_error_does_not_expose_valid_keys(
        self, client: TestClient, invalid_auth_headers: dict
    ):
        """Invalid key error should not expose valid keys."""
        response = client.get("/api/v1/video/list", headers=invalid_auth_headers)
        data = response.json()
        # Should not contain any valid keys
        response_str = str(data)
        assert VALID_API_KEY not in response_str
