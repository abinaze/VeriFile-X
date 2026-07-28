"""Tests for API key management and RBAC."""
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def temp_keys_file(tmp_path):
    temp = tmp_path / "test_api_keys.jsonl"
    with patch("backend.services.api_key_manager.KEYS_PATH", temp):
        yield temp


def test_create_key_returns_raw_key():
    from backend.services.api_key_manager import create_key
    result = create_key("Test Key", role="analyst")
    assert result["key"].startswith("vfx_")
    assert "warning" in result
    assert result["role"] == "analyst"

def test_create_key_requires_name():
    from backend.services.api_key_manager import create_key
    assert "error" in create_key("")

def test_create_key_invalid_role():
    from backend.services.api_key_manager import create_key
    assert "error" in create_key("Test", role="superuser")

def test_verify_valid_key():
    from backend.services.api_key_manager import create_key, verify_key
    created = create_key("Verify Test")
    result  = verify_key(created["key"])
    assert result is not None
    assert result["role"] == "analyst"
    assert "key_hash" not in result

def test_verify_invalid_key():
    from backend.services.api_key_manager import verify_key
    assert verify_key("vfx_notvalid") is None
    assert verify_key("wrong_prefix") is None
    assert verify_key("") is None

def test_verify_increments_use_count():
    from backend.services.api_key_manager import create_key, verify_key
    created = create_key("Counter Test")
    for _ in range(3):
        verify_key(created["key"])
    result = verify_key(created["key"])
    assert result["use_count"] >= 4

def test_revoke_key():
    from backend.services.api_key_manager import create_key, revoke_key, verify_key
    created = create_key("Revoke Test")
    revoke_key(created["key_id"])
    assert verify_key(created["key"]) is None

def test_list_keys():
    from backend.services.api_key_manager import create_key, list_keys
    create_key("Key A", role="viewer")
    create_key("Key B", role="analyst")
    keys = list_keys()
    assert len(keys) == 2
    for k in keys:
        assert "key_hash" not in k

def test_list_keys_excludes_revoked():
    from backend.services.api_key_manager import create_key, revoke_key, list_keys
    c1 = create_key("Active")
    c2 = create_key("Revoked")
    revoke_key(c2["key_id"])
    active = list_keys(include_inactive=False)
    assert len(active) == 1
    assert active[0]["key_id"] == c1["key_id"]


# ── F-6: real per-key salting, with backward compatibility ─────────────────

def test_new_keys_are_salted():
    """create_key() must generate a real, non-empty, per-key salt, and it
    must never appear in any response (same treatment as key_hash)."""
    from backend.services.api_key_manager import create_key, _load_keys
    created = create_key("Salt Test")
    assert "salt" not in created or created["salt"] == "[hidden]"
    stored = _load_keys()[created["key_id"]]
    assert stored.get("salt"), "expected a non-empty salt to be persisted"
    assert len(stored["salt"]) >= 16


def test_two_keys_get_different_salts():
    from backend.services.api_key_manager import create_key, _load_keys
    a = create_key("Salt A")
    b = create_key("Salt B")
    keys = _load_keys()
    assert keys[a["key_id"]]["salt"] != keys[b["key_id"]]["salt"]


def test_legacy_unsalted_key_still_verifies():
    """A record written before F-6 (no "salt" field, hash = sha256(raw_key)
    with no salt prefix) must keep verifying -- this is the whole point of
    defaulting entry.get("salt", "") to an empty string rather than
    requiring a migration of already-issued keys."""
    import hashlib
    from backend.services.api_key_manager import verify_key, _save_key, _now

    raw_key = "vfx_legacy_test_key_1234567890"
    legacy_entry = {
        "key_id": "legacy-1", "name": "Legacy Key", "description": "",
        "role": "analyst",
        "key_hash": hashlib.sha256(raw_key.encode("utf-8")).hexdigest(),  # no salt
        "created_at": _now(), "created_by": "system",
        "last_used": None, "use_count": 0, "active": True,
    }
    _save_key(legacy_entry)

    result = verify_key(raw_key)
    assert result is not None, "legacy unsalted key must still verify after F-6"
    assert result["role"] == "analyst"
    assert "salt" not in result
    assert "key_hash" not in result


def test_salt_never_leaks_from_verify_or_list():
    from backend.services.api_key_manager import create_key, verify_key, list_keys
    created = create_key("No Leak Test")
    result  = verify_key(created["key"])
    assert "salt" not in result
    for k in list_keys():
        assert "salt" not in k




def test_api_create_key_without_auth():
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)  # deliberately unauthenticated — this test checks the no-auth path
    response = client.post("/api/v1/keys/", json={"name": "No Auth", "role": "analyst"})
    assert response.status_code == 401

def test_api_verify_endpoint_valid(client):
    from backend.services.api_key_manager import create_key
    created = create_key("API Test", role="analyst")
    response = client.get("/api/v1/keys/verify",
                          headers={"Authorization": f"Bearer {created['key']}"})
    assert response.status_code == 200
    assert response.json()["valid"] is True
    assert response.json()["role"] == "analyst"

def test_api_verify_endpoint_invalid(client):
    response = client.get("/api/v1/keys/verify",
                          headers={"Authorization": "Bearer vfx_invalid"})
    assert response.status_code == 401
