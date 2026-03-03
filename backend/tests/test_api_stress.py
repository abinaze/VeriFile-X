"""
API stress and edge case tests.
"""
import pytest


def test_upload_non_image_file(client):
    """Test uploading a non-image file."""
    files = {"file": ("test.txt", b"This is text", "text/plain")}
    response = client.post("/api/v1/analyze/image", files=files)
    assert response.status_code in [400, 422]


def test_upload_empty_file(client):
    """Test uploading an empty file."""
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/api/v1/analyze/image", files=files)
    assert response.status_code in [400, 422]


def test_api_response_schema(client, sample_image_bytes):
    """Validate API response structure."""
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    response = client.post("/api/v1/analyze/image", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    required_keys = ["metadata", "file_info", "hashes", "ai_detection", "summary"]
    for key in required_keys:
        assert key in data


def test_health_endpoint_response(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
