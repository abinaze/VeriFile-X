"""
Webhook Delivery Service — Phase 19.

Registers outbound webhooks, signs payloads with HMAC-SHA256,
retries failed deliveries (3×: 5s / 30s / 120s), suspends
endpoints that exceed the failure threshold, and logs every
delivery attempt to webhooks_deliveries.jsonl.

Security design:
  - Raw secret shown ONCE at registration, then discarded.
  - Stored value is SHA-256(secret) — used as the HMAC signing key.
    This is a documented trade-off; encrypted storage is Phase 30.
  - All JSONL writes are inside a threading.Lock().
"""
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen, Request as _URLRequest
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

# ── Storage paths (always relative to this file, never CWD) ─────────────────
_DATA_DIR = Path(__file__).parent.parent / "data"
_HOOKS_PATH = _DATA_DIR / "webhooks.jsonl"
_DELIV_PATH = _DATA_DIR / "webhook_deliveries.jsonl"

_hooks_lock = threading.Lock()
_deliv_lock = threading.Lock()

# After this many consecutive failures the webhook is suspended.
_SUSPEND_AFTER = 3

# Retry back-off delays in seconds.
_RETRY_DELAYS = (5, 30, 120)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ensure_dirs() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_hooks() -> Dict[str, Dict[str, Any]]:
    """Return {webhook_id: entry} — last record per ID wins."""
    hooks: Dict[str, Dict[str, Any]] = {}
    if not _HOOKS_PATH.exists():
        return hooks
    try:
        for line in _HOOKS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                wid = entry.get("webhook_id")
                if wid:
                    hooks[wid] = entry
            except json.JSONDecodeError:
                continue
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load webhooks: %s", exc, exc_info=True)
    return hooks


def _append_hook(entry: Dict[str, Any]) -> None:
    _ensure_dirs()
    with _hooks_lock:
        with _HOOKS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def _append_delivery(entry: Dict[str, Any]) -> None:
    _ensure_dirs()
    with _deliv_lock:
        with _DELIV_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


# ── Public API ───────────────────────────────────────────────────────────────

def register_webhook(
    url: str,
    name: str,
    events: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Register a new webhook endpoint.

    Returns the entry **including the raw secret** (shown once only).
    The caller is responsible for storing the secret; it is not recoverable.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    webhook_id = str(uuid.uuid4())
    # Log BEFORE generating the signing token so the token never appears
    # in scope at any logger call — prevents CodeQL CWE-312 taint path.
    logger.info("Registering webhook %s -> %s", webhook_id, url)

    # One-time registration token — shown to caller once, never logged.
    _token = secrets.token_hex(32)
    secret_hash = _sha256(_token)

    entry: Dict[str, Any] = {
        "webhook_id":     webhook_id,
        "url":            url,
        "name":           name,
        "events":         events or ["analysis.complete"],
        "secret_hash":    secret_hash,
        "active":         True,
        "status":         "active",
        "delivery_count": 0,
        "failure_count":  0,
        "created_at":     _now(),
    }
    _append_hook(entry)

    # Build response with one-time token then immediately discard local ref.
    response = dict(entry)
    response["secret"] = _token  # noqa: S105 -- intentional single-use exposure
    del _token
    return response


def list_webhooks() -> List[Dict[str, Any]]:
    """Return all non-deleted webhooks (secret_hash redacted)."""
    hooks = _load_hooks()
    result = []
    for h in hooks.values():
        if h.get("status") == "deleted":
            continue
        safe = {k: v for k, v in h.items() if k != "secret_hash"}
        result.append(safe)
    return result


def delete_webhook(webhook_id: str) -> bool:
    """Soft-delete a webhook (sets status=deleted, active=False)."""
    hooks = _load_hooks()
    if webhook_id not in hooks:
        return False
    entry = {**hooks[webhook_id], "status": "deleted", "active": False}
    _append_hook(entry)
    logger.info("Deleted webhook id ending with %s", webhook_id[-8:])
    return True


def get_deliveries(webhook_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent delivery log entries, optionally filtered by webhook_id."""
    if not _DELIV_PATH.exists():
        return []
    try:
        lines = _DELIV_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    entries = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if webhook_id and entry.get("webhook_id") != webhook_id:
            continue
        entries.append(entry)
        if len(entries) >= limit:
            break
    return entries


def sign_payload(secret_hash: str, body: bytes) -> str:
    """Return HMAC-SHA256 hex digest using the stored secret_hash as the key."""
    return hmac.new(
        secret_hash.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()


def send_test(webhook_id: str) -> Dict[str, Any]:
    """Queue a test delivery for an existing webhook (non-blocking)."""
    hooks = _load_hooks()
    hook = hooks.get(webhook_id)
    if not hook or hook.get("status") == "deleted":
        raise KeyError(f"Webhook {webhook_id!r} not found")

    test_payload = {
        "event":      "test",
        "webhook_id": webhook_id,
        "timestamp":  _now(),
        "message":    "VeriFile-X test delivery",
    }
    t = threading.Thread(
        target=_deliver_with_retry,
        args=(hook, "test", test_payload),
        daemon=True,
    )
    t.start()
    return {"queued": True, "webhook_id": webhook_id}


def fire_webhooks(
    evidence_id: str,
    result: Dict[str, Any],
    event: str = "analysis.complete",
) -> int:
    """
    Fire outbound deliveries to all active webhooks subscribed to *event*.

    Each delivery runs in a daemon thread (fire-and-forget).
    Returns the number of webhooks targeted.
    """
    hooks = _load_hooks()
    targeted = 0
    for hook in hooks.values():
        if not hook.get("active"):
            continue
        if hook.get("status") != "active":
            continue
        if event not in hook.get("events", []):
            continue
        payload = {
            "event":       event,
            "evidence_id": evidence_id,
            "webhook_id":  hook["webhook_id"],
            "timestamp":   _now(),
            "result":      result,
        }
        t = threading.Thread(
            target=_deliver_with_retry,
            args=(hook, event, payload),
            daemon=True,
        )
        t.start()
        targeted += 1
    logger.info("fire_webhooks: targeted %d webhook(s) for event=%s", targeted, event)
    return targeted


# ── Internal delivery logic ───────────────────────────────────────────────────

def _deliver_with_retry(
    hook: Dict[str, Any],
    event: str,
    payload: Dict[str, Any],
) -> None:
    """Attempt delivery with up to 3 retries; suspend hook on sustained failure."""
    delivery_id = str(uuid.uuid4())
    body = json.dumps(payload).encode("utf-8")
    sig = sign_payload(hook["secret_hash"], body)

    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        success, status_code, error_msg = _attempt_delivery(
            hook["url"], body, sig, hook["webhook_id"], delivery_id
        )
        _log_delivery(
            delivery_id=delivery_id,
            webhook_id=hook["webhook_id"],
            event=event,
            attempt=attempt,
            success=success,
            status_code=status_code,
            error=error_msg,
        )
        if success:
            _update_hook_counts(hook["webhook_id"], success=True)
            return

        # Count every failed attempt so failure_count accumulates correctly
        # across all retries. _update_hook_counts suspends when >= _SUSPEND_AFTER.
        _update_hook_counts(hook["webhook_id"], success=False)

        if attempt < len(_RETRY_DELAYS):
            logger.warning(
                "Webhook %s delivery failed (attempt %d/%d), retrying in %ds",
                hook["webhook_id"], attempt, len(_RETRY_DELAYS), delay,
            )
            time.sleep(delay)

    logger.error("Webhook %s exhausted all retries", hook["webhook_id"])


def _attempt_delivery(
    url: str,
    body: bytes,
    sig: str,
    webhook_id: str,
    delivery_id: str,
) -> tuple[bool, Optional[int], Optional[str]]:
    """Perform a single HTTP POST. Returns (success, status_code, error)."""
    req = _URLRequest(
        url,
        data=body,
        headers={
            "Content-Type":        "application/json",
            "X-VeriFile-Signature": f"sha256={sig}",
            "X-VeriFile-Event":    webhook_id,
            "X-Delivery-ID":       delivery_id,
            "User-Agent":          "VeriFile-X-Webhook/7.3.0",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:  # noqa: S310
            return True, resp.status, None
    except HTTPError as exc:
        return False, exc.code, str(exc)
    except URLError as exc:
        return False, None, str(exc.reason)
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def _log_delivery(
    *,
    delivery_id: str,
    webhook_id: str,
    event: str,
    attempt: int,
    success: bool,
    status_code: Optional[int],
    error: Optional[str],
) -> None:
    entry = {
        "delivery_id": delivery_id,
        "webhook_id":  webhook_id,
        "event":       event,
        "attempt":     attempt,
        "success":     success,
        "status_code": status_code,
        "error":       error,
        "timestamp":   _now(),
    }
    _append_delivery(entry)


def _update_hook_counts(webhook_id: str, *, success: bool) -> None:
    """Append an updated hook record reflecting delivery outcome."""
    hooks = _load_hooks()
    hook = hooks.get(webhook_id)
    if not hook:
        return
    updated = {**hook}
    if success:
        updated["delivery_count"] = hook.get("delivery_count", 0) + 1
        updated["failure_count"] = 0  # Reset streak on success
    else:
        new_failures = hook.get("failure_count", 0) + 1
        updated["failure_count"] = new_failures
        if new_failures >= _SUSPEND_AFTER:
            updated["status"] = "suspended"
            updated["active"] = False
            logger.warning(
                "Webhook %s suspended after %d consecutive failures",
                webhook_id, new_failures,
            )
    _append_hook(updated)
