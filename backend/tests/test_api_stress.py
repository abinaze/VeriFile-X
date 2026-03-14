"""
API stress testing - edge cases and error handling.

Tests validate proper HTTP status codes and error responses.
"""
import pytest


def test_upload_non_image_file(client):
    """
    Test rejection of non-image files.
    
    Backend returns 415 (Unsupported Media Type) for non-image uploads.
    This is the semantically correct HTTP status code per RFC 7231.
    """
    file_content = b"This is not an image"
    files = {"file": ("test.txt", file_content, "text/plain")}
    
    response = client.post("/api/v1/analyze/image", files=files)
    
    # Accept 400, 415, or 422 (backend uses 415 for semantic correctness)
    assert response.status_code in [400, 415, 422], (
        f"Expected 400/415/422 for non-image, got {response.status_code}"
    )


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
