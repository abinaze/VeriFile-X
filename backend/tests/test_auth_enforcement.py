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


# ── F-4: auth centralization must not silently re-diverge ──────────────────
#
# feedback.py, keys.py, and webhooks.py used to each define their own
# _require_admin/_require_analyst -- byte-for-byte similar to
# backend/core/auth.py's versions at first, but free to drift the moment
# any one copy was edited without the others (which is exactly what
# happened: keys.py's copy silently became stricter about the "Bearer "
# prefix than the other four). Checking that the call sites *import the
# same function object* (not just "equivalent-looking code") is the only
# way to make that class of drift structurally impossible instead of
# merely unlikely.

def test_admin_auth_is_the_same_shared_object_everywhere():
    import backend.core.auth as core_auth
    import backend.api.routes.keys as keys_route
    import backend.api.routes.webhooks as webhooks_route
    import backend.api.routes.feedback as feedback_route

    assert keys_route._require_admin is core_auth.require_admin
    assert webhooks_route._require_admin is core_auth.require_admin
    assert feedback_route._require_admin is core_auth.require_admin


def test_analyst_auth_is_the_same_shared_object_everywhere():
    import backend.core.auth as core_auth
    import backend.api.routes.feedback as feedback_route

    assert feedback_route._require_analyst is core_auth.require_analyst


def test_analyze_rejects_invalid_key():
    client = _bare_client()
    client.headers.update({"Authorization": "Bearer vfx_not_a_real_key"})
    files = {"file": ("test.png", b"not a real image", "image/png")}
    response = client.post("/api/v1/analyze/image", files=files)
    assert response.status_code == 401
