"""
Tests for API endpoints.
"""
import pytest


def test_root_endpoint(client):
    """Test root endpoint returns frontend HTML or API info."""
    response = client.get("/")
    assert response.status_code == 200
    # Root now serves HTML frontend, so check content type
    assert "text/html" in response.headers.get("content-type", "") or response.json()


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")  # degraded when model files absent (CI)
    assert "timestamp" in data
