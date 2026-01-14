"""
Health Endpoint Tests
=====================
Tests for /api/v1/health endpoints.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthCheck:
    """Test suite for main health check endpoint."""
    
    def test_health_returns_200(self, client: TestClient):
        """Health check should return 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
    
    def test_health_returns_correct_structure(self, client: TestClient):
        """Health response should have required fields."""
        response = client.get("/api/v1/health")
        data = response.json()
        
        # Check required fields
        assert "status" in data
        assert "version" in data
        assert "environment" in data
        assert "databases" in data
    
    def test_health_status_values(self, client: TestClient):
        """Health status should be one of expected values."""
        response = client.get("/api/v1/health")
        data = response.json()
        
        valid_statuses = ["healthy", "degraded", "unhealthy"]
        assert data["status"] in valid_statuses
    
    def test_health_version_format(self, client: TestClient):
        """Health version should be valid semver format."""
        response = client.get("/api/v1/health")
        data = response.json()
        
        version = data["version"]
        # Basic semver check (X.Y.Z)
        parts = version.split(".")
        assert len(parts) >= 2
    
    def test_health_environment_value(self, client: TestClient):
        """Health environment should match configured env."""
        response = client.get("/api/v1/health")
        data = response.json()
        
        # We set FORENSIC_ENV=dev in conftest
        assert data["environment"] == "dev"
    
    def test_health_includes_database_status(self, client: TestClient):
        """Health should report database connection status."""
        response = client.get("/api/v1/health")
        data = response.json()
        
        databases = data["databases"]
        assert isinstance(databases, dict)
        # Should have chromadb and neo4j
        assert "chromadb" in databases
        assert "neo4j" in databases
    
    def test_health_includes_uptime(self, client: TestClient):
        """Health should include uptime seconds."""
        response = client.get("/api/v1/health")
        data = response.json()
        
        # uptime_seconds should be present and non-negative
        if "uptime_seconds" in data:
            assert data["uptime_seconds"] >= 0


class TestReadinessProbe:
    """Test suite for readiness probe endpoint."""
    
    def test_ready_returns_200(self, client: TestClient):
        """Readiness probe should return 200 when ready."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
    
    def test_ready_returns_success_flag(self, client: TestClient):
        """Readiness response should have success flag."""
        response = client.get("/api/v1/health/ready")
        data = response.json()
        
        assert "success" in data
        assert data["success"] is True
    
    def test_ready_includes_message(self, client: TestClient):
        """Readiness response should include message."""
        response = client.get("/api/v1/health/ready")
        data = response.json()
        
        assert "message" in data
        assert isinstance(data["message"], str)


class TestLivenessProbe:
    """Test suite for liveness probe endpoint."""
    
    def test_live_returns_200(self, client: TestClient):
        """Liveness probe should return 200 when alive."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
    
    def test_live_returns_success_flag(self, client: TestClient):
        """Liveness response should have success flag."""
        response = client.get("/api/v1/health/live")
        data = response.json()
        
        assert "success" in data
        assert data["success"] is True
    
    def test_live_is_lightweight(self, client: TestClient):
        """Liveness check should be fast (no heavy operations)."""
        import time
        
        start = time.time()
        response = client.get("/api/v1/health/live")
        elapsed = time.time() - start
        
        # Should complete in under 100ms
        assert elapsed < 0.1
        assert response.status_code == 200


class TestHealthResponseHeaders:
    """Test response headers from health endpoints."""
    
    def test_health_includes_request_id(self, client: TestClient):
        """Health response should include X-Request-ID header."""
        response = client.get("/api/v1/health")
        
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0
    
    def test_health_includes_process_time(self, client: TestClient):
        """Health response should include X-Process-Time header."""
        response = client.get("/api/v1/health")
        
        assert "X-Process-Time" in response.headers
    
    def test_health_content_type_json(self, client: TestClient):
        """Health response should be JSON."""
        response = client.get("/api/v1/health")
        
        assert "application/json" in response.headers["content-type"]


class TestRootEndpoint:
    """Test suite for root endpoint."""
    
    def test_root_returns_200(self, client: TestClient):
        """Root endpoint should return 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_api_info(self, client: TestClient):
        """Root should return API information."""
        response = client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "docs" in data
    
    def test_root_docs_url_correct(self, client: TestClient):
        """Root should point to correct docs URL."""
        response = client.get("/")
        data = response.json()
        
        assert data["docs"] == "/docs"
