"""
Shared test fixtures and configuration.
Why: Reusable test setup across all test files.
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    """
    TestClient fixture for API testing.
    Why: Provides synchronous client for easy testing.
    """
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """
    Mock image file bytes for testing.
    Why: Testing file uploads without real files.
    """
    # 1x1 PNG image (smallest valid PNG)
    return (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
        b'\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
        b'\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )
