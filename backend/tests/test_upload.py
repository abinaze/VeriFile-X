"""
Tests for file upload endpoint.
"""
import pytest
from io import BytesIO


def test_validate_upload_success(client, sample_image_bytes):
    """Test successful file validation."""
    files = {"file": ("test.png", BytesIO(sample_image_bytes), "image/png")}
    
    response = client.post("/api/v1/upload/validate", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["mime_type"] == "image/png"
    assert data["filename"] == "test.png"


def test_validate_upload_invalid_type(client):
    """Test upload with invalid file type."""
    files = {"file": ("test.txt", BytesIO(b"Plain text"), "text/plain")}
    
    response = client.post("/api/v1/upload/validate", files=files)
    
    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"]


def test_validate_upload_too_large(client):
    """Test upload exceeding size limit."""
    # Create 60MB file
    large_file = BytesIO(b"x" * (60 * 1024 * 1024))
    files = {"file": ("huge.bin", large_file, "application/octet-stream")}
    
    response = client.post("/api/v1/upload/validate", files=files)
    
    assert response.status_code == 400
    assert "exceeds limit" in response.json()["detail"]
