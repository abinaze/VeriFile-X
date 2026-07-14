"""
Explicit tests that analyze.py and cases.py reject unauthenticated
requests. Added alongside the auth fix itself so the fix can never
silently regress — a prior audit found these endpoints had zero such
coverage, which is exactly how they stayed open for as long as they did.
"""
from fastapi.testclient import TestClient
from backend.main import app


def _bare_client():
    """A client with no Authorization header — for testing rejection."""
    return TestClient(app)


def test_analyze_image_rejects_no_auth():
    client = _bare_client()
    files = {"file": ("test.png", b"not a real image", "image/png")}
    response = client.post("/api/v1/analyze/image", files=files)
    assert response.status_code == 401


def test_analyze_history_rejects_no_auth():
    client = _bare_client()
    response = client.get("/api/v1/analyze/history")
    assert response.status_code == 401


def test_cases_create_rejects_no_auth():
    client = _bare_client()
    response = client.post("/api/v1/cases/", json={"name": "test"})
    assert response.status_code == 401


def test_cases_list_rejects_no_auth():
    client = _bare_client()
    response = client.get("/api/v1/cases/")
    assert response.status_code == 401


def test_analyze_rejects_invalid_key():
    client = _bare_client()
    client.headers.update({"Authorization": "Bearer vfx_not_a_real_key"})
    files = {"file": ("test.png", b"not a real image", "image/png")}
    response = client.post("/api/v1/analyze/image", files=files)
    assert response.status_code == 401
