"""
Basic API endpoint tests.
Why: Catch regressions before they reach production.
"""
from fastapi.testclient import TestClient
from backend.main import app

# TestClient makes synchronous requests (easier for testing)
client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["name"] == "VeriFile-X API"
    assert data["status"] == "operational"
    assert "/docs" in data["docs"]


def test_health_check():
    """Test health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "debug_mode" in data
