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
    """Test upload exceeding size limit.

    F-15: this now gets rejected EARLY via the Content-Length pre-check
    (413) rather than being fully read into memory first and rejected
    post-read (previously 400) -- 413 Payload Too Large is also the more
    semantically correct status code, and matches what
    /api/v1/analyze/image's equivalent pre-check already returns.
    """
    # Create 60MB file
    large_file = BytesIO(b"x" * (60 * 1024 * 1024))
    files = {"file": ("huge.bin", large_file, "application/octet-stream")}
    
    response = client.post("/api/v1/upload/validate", files=files)
    
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_validate_file_post_read_check_still_enforces_limit_independently():
    """F-15 regression test: the Content-Length pre-check only helps
    when the header is present and accurate (absent under chunked
    transfer, or a client could send an inaccurate one). validate_file()
    itself -- the post-read check this endpoint also calls -- must
    independently still reject an oversized file regardless of the
    pre-check, so there's real defense in depth rather than the new
    check silently replacing the old one.
    """
    from backend.utils.validators import validate_file, FileValidationError
    with pytest.raises(FileValidationError, match="exceeds limit"):
        validate_file(b"x" * (60 * 1024 * 1024), "huge.bin")
