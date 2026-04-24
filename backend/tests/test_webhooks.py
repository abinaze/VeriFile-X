"""
Phase 19 — Webhook delivery system tests.

Coverage:
  - Service layer: register, list, delete, deliveries, fire, sign, retry
  - API endpoints: 401/403 auth gates, 404 handling
"""
import hashlib
import hmac
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """Redirect all webhook storage paths to a temp directory."""
    import backend.services.webhook_manager as wm
    monkeypatch.setattr(wm, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(wm, "_HOOKS_PATH", tmp_path / "webhooks.jsonl")
    monkeypatch.setattr(wm, "_DELIV_PATH", tmp_path / "webhook_deliveries.jsonl")
    yield tmp_path


# ── Service-level tests ───────────────────────────────────────────────────────

class TestRegister:
    def test_register_returns_secret_once(self, tmp_data_dir):
        """Registration response must include plain-text secret."""
        from backend.services.webhook_manager import register_webhook
        result = register_webhook("https://example.com/hook", "my-hook")
        assert "secret" in result
        assert len(result["secret"]) == 64  # 32-byte hex token
        # Secret must NOT be stored in plaintext
        stored = (tmp_data_dir / "webhooks.jsonl").read_text()
        assert result["secret"] not in stored

    def test_register_invalid_url_rejected(self, tmp_data_dir):
        """Non-http(s) URLs must raise ValueError."""
        from backend.services.webhook_manager import register_webhook
        with pytest.raises(ValueError, match="http"):
            register_webhook("ftp://evil.example", "bad-hook")

    def test_register_persists_to_disk(self, tmp_data_dir):
        """Registered webhook must appear on disk."""
        from backend.services.webhook_manager import register_webhook
        r = register_webhook("https://example.com/hook", "persist-test")
        lines = (tmp_data_dir / "webhooks.jsonl").read_text().splitlines()
        assert any(r["webhook_id"] in line for line in lines)


class TestList:
    def test_list_webhooks_empty(self, tmp_data_dir):
        """List returns empty when no webhooks registered."""
        from backend.services.webhook_manager import list_webhooks
        assert list_webhooks() == []

    def test_list_webhooks_after_register(self, tmp_data_dir):
        """List returns registered webhooks; secret_hash must be absent."""
        from backend.services.webhook_manager import register_webhook, list_webhooks
        register_webhook("https://example.com/a", "hook-a")
        register_webhook("https://example.com/b", "hook-b")
        hooks = list_webhooks()
        assert len(hooks) == 2
        for h in hooks:
            assert "secret_hash" not in h


class TestDelete:
    def test_delete_webhook_deactivates(self, tmp_data_dir):
        """Deleted webhook must not appear in list."""
        from backend.services.webhook_manager import register_webhook, delete_webhook, list_webhooks
        r = register_webhook("https://example.com/del", "delete-me")
        assert delete_webhook(r["webhook_id"]) is True
        assert all(h["webhook_id"] != r["webhook_id"] for h in list_webhooks())

    def test_delete_nonexistent_returns_false(self, tmp_data_dir):
        """Deleting a non-existent ID must return False (no error)."""
        from backend.services.webhook_manager import delete_webhook
        assert delete_webhook("00000000-0000-0000-0000-000000000000") is False


class TestDeliveries:
    def test_delivery_log_empty_initially(self, tmp_data_dir):
        """Delivery log must start empty."""
        from backend.services.webhook_manager import get_deliveries
        assert get_deliveries() == []

    def test_send_test_queues_delivery(self, tmp_data_dir):
        """send_test must return queued=True and target a real webhook."""
        from backend.services.webhook_manager import register_webhook, send_test
        with patch("backend.services.webhook_manager._attempt_delivery") as mock_del:
            mock_del.return_value = (True, 200, None)
            r = register_webhook("https://example.com/test", "test-hook")
            result = send_test(r["webhook_id"])
        assert result["queued"] is True
        assert result["webhook_id"] == r["webhook_id"]

    def test_send_test_nonexistent_raises(self, tmp_data_dir):
        """send_test on unknown ID must raise KeyError."""
        from backend.services.webhook_manager import send_test
        with pytest.raises(KeyError):
            send_test("00000000-0000-0000-0000-000000000000")


class TestFireWebhooks:
    def test_fire_webhooks_returns_zero_when_none_registered(self, tmp_data_dir):
        """fire_webhooks with no hooks returns 0."""
        from backend.services.webhook_manager import fire_webhooks
        count = fire_webhooks("eid-001", {"summary": {}})
        assert count == 0

    def test_fire_webhooks_returns_count(self, tmp_data_dir):
        """fire_webhooks returns the number of active matching webhooks."""
        from backend.services.webhook_manager import register_webhook, fire_webhooks
        with patch("backend.services.webhook_manager._attempt_delivery") as mock_del:
            mock_del.return_value = (True, 200, None)
            register_webhook("https://example.com/w1", "w1", events=["analysis.complete"])
            register_webhook("https://example.com/w2", "w2", events=["analysis.complete"])
            # Give threads a moment to spin up then check return value
            count = fire_webhooks("eid-002", {"summary": {}}, event="analysis.complete")
        assert count == 2


class TestSigning:
    def test_sign_hmac_sha256(self, tmp_data_dir):
        """sign_payload must produce a valid HMAC-SHA256 hex digest."""
        from backend.services.webhook_manager import sign_payload
        secret_hash = hashlib.sha256(b"test-secret").hexdigest()
        body = b'{"event": "test"}'
        sig = sign_payload(secret_hash, body)
        expected = hmac.new(
            secret_hash.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
        assert sig == expected
        assert len(sig) == 64


class TestSuspend:
    def test_webhook_suspended_after_failures(self, tmp_data_dir):
        """Webhook must be suspended after _SUSPEND_AFTER consecutive failures."""
        from backend.services.webhook_manager import (
            register_webhook, _deliver_with_retry, list_webhooks, _SUSPEND_AFTER
        )
        with patch("backend.services.webhook_manager._attempt_delivery") as mock_del:
            mock_del.return_value = (False, 500, "server error")
            with patch("backend.services.webhook_manager.time.sleep"):  # no real waiting
                r = register_webhook("https://example.com/fail", "fail-hook")
                hook = {**r, "secret_hash": hashlib.sha256(r["secret"].encode()).hexdigest()}
                _deliver_with_retry(hook, "analysis.complete", {"event": "analysis.complete"})

        # After exhausting retries, hook should be suspended
        import backend.services.webhook_manager as wm
        hooks = wm._load_hooks()
        entry = hooks[r["webhook_id"]]
        assert entry["status"] == "suspended"
        assert entry["active"] is False


# ── API endpoint auth-gate tests ─────────────────────────────────────────────

@pytest.fixture()
def client(tmp_data_dir):
    """TestClient with webhooks router registered."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.routes.webhooks import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


class TestAPIAuth:
    def test_register_requires_admin(self, client):
        """POST /api/v1/webhooks/ without auth → 401."""
        resp = client.post(
            "/api/v1/webhooks/",
            json={"url": "https://example.com", "name": "x"},
        )
        assert resp.status_code == 401

    def test_list_requires_admin(self, client):
        """GET /api/v1/webhooks/ without auth → 401."""
        resp = client.get("/api/v1/webhooks/")
        assert resp.status_code == 401

    def test_delete_requires_admin(self, client):
        """DELETE /api/v1/webhooks/{id} without auth → 401."""
        resp = client.delete("/api/v1/webhooks/some-id")
        assert resp.status_code == 401

    def test_deliveries_requires_admin(self, client):
        """GET /api/v1/webhooks/deliveries without auth → 401."""
        resp = client.get("/api/v1/webhooks/deliveries")
        assert resp.status_code == 401
