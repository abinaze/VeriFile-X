"""
API stress testing - edge cases and error handling.

Tests validate proper HTTP status codes and error responses.
"""
import pytest


def test_upload_non_image_file(client):
    """
    Test rejection of non-image files.
    
    Expected: 415 Unsupported Media Type (semantic HTTP status code)
    Updated: Changed from 400 to 415 for semantic accuracy
    """
    file_content = b"This is not an image"
    files = {"file": ("test.txt", file_content, "text/plain")}
    
    response = client.post("/api/v1/analyze/image", files=files)
    
    # 415 = Unsupported Media Type (RFC 7231)
    assert response.status_code == 415, (
        f"Expected 415 (Unsupported Media Type) for non-image file, "
        f"got {response.status_code}"
    )
    
    # Verify error message is informative
    data = response.json()
    assert "detail" in data
    assert "Unsupported" in data["detail"] or "media type" in data["detail"].lower()


def test_upload_empty_file(client):
    """Test handling of empty file uploads."""
    files = {"file": ("empty.png", b"", "image/png")}
    
    response = client.post("/api/v1/analyze/image", files=files)
    
    # Empty file should fail validation
    assert response.status_code in [400, 422], (
        f"Expected 400 or 422 for empty file, got {response.status_code}"
    )


def test_api_response_schema(client, sample_image_bytes):
    """
    Test that API responses follow expected schema.
    
    Validates presence of required fields in successful response.
    """
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    
    response = client.post("/api/v1/analyze/image", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify top-level structure
    assert "metadata" in data
    assert "file_info" in data
    assert "hashes" in data
    assert "summary" in data
    
    # Verify summary has required fields
    summary = data["summary"]
    assert "ai_probability" in summary
    assert "ai_classification" in summary
    assert "total_detection_signals" in summary
    
    # Verify metadata
    assert "analyzer_version" in data["metadata"]
    assert data["metadata"]["analyzer_version"] == "6.0.0"


def test_health_endpoint_response(client):
    """Test health check endpoint returns expected structure."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert data["status"] in ["healthy", "ok", "ready"]
