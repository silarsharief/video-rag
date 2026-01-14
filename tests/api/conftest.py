"""
Pytest Fixtures for API Tests
=============================
Shared test configuration and fixtures used across all API tests.
"""

import os
import sys

# CRITICAL: Set environment variables BEFORE any imports
# This must happen before config/settings.py is imported
os.environ["FORENSIC_ENV"] = "dev"
os.environ["SKIP_CONFIG_VALIDATION"] = "true"
os.environ["API_KEYS"] = "test-api-key-12345,another-test-key"

import tempfile
import shutil
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# TEST CONSTANTS
# ============================================================================
VALID_API_KEY = "test-api-key-12345"
ANOTHER_VALID_KEY = "another-test-key"
INVALID_API_KEY = "invalid-key-xyz"


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def test_api_key() -> str:
    """Provide valid API key for authenticated requests."""
    return VALID_API_KEY


@pytest.fixture(scope="session")
def invalid_api_key() -> str:
    """Provide invalid API key for auth failure tests."""
    return INVALID_API_KEY


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """
    Create FastAPI test client.
    Scope: module (shared across tests in same file)
    """
    # Patch settings before importing app
    import config.settings as settings
    settings.API_KEYS = [VALID_API_KEY, ANOTHER_VALID_KEY]
    
    # Use real video storage dir for tests (data/videos/)
    # This allows testing with existing videos
    from pathlib import Path
    settings.VIDEO_STORAGE_DIR = Path(project_root) / "data" / "videos"
    settings.VIDEO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    from src.api.main import app
    
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def temp_video_dir() -> Generator[Path, None, None]:
    """
    Create temporary directory for video uploads.
    Cleaned up after each test.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="test_videos_"))
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_video_file(temp_video_dir: Path) -> Path:
    """
    Create a minimal valid MP4 file for testing.
    This is a tiny valid MP4 header (not playable but valid format).
    """
    # Minimal MP4 file bytes (ftyp + moov atoms)
    # This is enough to pass file validation but not actually playable
    mp4_header = bytes([
        # ftyp atom
        0x00, 0x00, 0x00, 0x14,  # size: 20 bytes
        0x66, 0x74, 0x79, 0x70,  # type: 'ftyp'
        0x69, 0x73, 0x6F, 0x6D,  # brand: 'isom'
        0x00, 0x00, 0x00, 0x01,  # version
        0x69, 0x73, 0x6F, 0x6D,  # compatible brand
        # moov atom (minimal)
        0x00, 0x00, 0x00, 0x08,  # size: 8 bytes
        0x6D, 0x6F, 0x6F, 0x76,  # type: 'moov'
    ])
    
    video_path = temp_video_dir / "test_video.mp4"
    video_path.write_bytes(mp4_header)
    return video_path


@pytest.fixture(scope="function")
def sample_mov_file(temp_video_dir: Path) -> Path:
    """Create a minimal MOV file for testing."""
    # MOV uses same container format as MP4
    mov_header = bytes([
        0x00, 0x00, 0x00, 0x14,
        0x66, 0x74, 0x79, 0x70,
        0x71, 0x74, 0x20, 0x20,  # 'qt  ' brand
        0x00, 0x00, 0x00, 0x00,
        0x71, 0x74, 0x20, 0x20,
        0x00, 0x00, 0x00, 0x08,
        0x6D, 0x6F, 0x6F, 0x76,
    ])
    
    video_path = temp_video_dir / "test_video.mov"
    video_path.write_bytes(mov_header)
    return video_path


@pytest.fixture(scope="function")
def invalid_file(temp_video_dir: Path) -> Path:
    """Create an invalid file (wrong extension) for testing."""
    file_path = temp_video_dir / "test_file.txt"
    file_path.write_text("This is not a video file")
    return file_path


@pytest.fixture(scope="function")
def auth_headers(test_api_key: str) -> dict:
    """Provide headers with valid API key."""
    return {"X-API-Key": test_api_key}


@pytest.fixture(scope="function")
def invalid_auth_headers(invalid_api_key: str) -> dict:
    """Provide headers with invalid API key."""
    return {"X-API-Key": invalid_api_key}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def assert_error_response(response, expected_status: int, expected_error_code: str):
    """
    Helper to assert error response structure.
    
    Args:
        response: FastAPI TestClient response
        expected_status: Expected HTTP status code
        expected_error_code: Expected error_code in response
    """
    assert response.status_code == expected_status
    data = response.json()
    assert "detail" in data or "error" in data
    if "detail" in data:
        detail = data["detail"]
        if isinstance(detail, dict):
            assert detail.get("error_code") == expected_error_code


def assert_success_response(response, expected_status: int = 200):
    """
    Helper to assert successful response.
    
    Args:
        response: FastAPI TestClient response
        expected_status: Expected HTTP status code
    """
    assert response.status_code == expected_status
    data = response.json()
    # Should not have error structure
    assert "error_code" not in data
