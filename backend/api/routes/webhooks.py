"""
Webhook management endpoints — admin only.

All endpoints require Authorization: Bearer <admin-key>.

Routes:
  POST   /api/v1/webhooks/          — register
  GET    /api/v1/webhooks/          — list
  DELETE /api/v1/webhooks/{id}      — soft-delete
  POST   /api/v1/webhooks/{id}/test — queue test delivery
  GET    /api/v1/webhooks/deliveries — delivery log
"""
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, HttpUrl
from typing import List, Optional

from slowapi import Limiter
from slowapi.util import get_remote_address
from backend.core.config import settings as _settings

# Rate limits every webhook endpoint (send_test in particular triggers
# an on-demand outbound HTTP request).
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{_settings.RATE_LIMIT_PER_MINUTE}/minute"])

router = APIRouter(prefix="/api/v1/webhooks", tags=["Webhooks"])


# ── Auth (F-4: now the real shared core.auth implementation, not a local copy) ──

from backend.core.auth import require_admin as _require_admin


# ── Request bodies ────────────────────────────────────────────────────────────

class WebhookRegisterRequest(BaseModel):
    url: HttpUrl
    name: str
    events: List[str] = ["analysis.complete"]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", summary="Register webhook (admin only)")
async def register_webhook(
    body: WebhookRegisterRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Register a new outbound webhook.

    Returns the entry including the **raw secret** — this is the only time
    it is shown.  Store it securely; it cannot be retrieved later.
    """
    _require_admin(authorization)
    from backend.services.webhook_manager import register_webhook as _register
    try:
        result = _register(
            url=str(body.url),
            name=body.name,
            events=body.events,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result


@router.get("/", summary="List webhooks (admin only)")
async def list_webhooks(
    authorization: Optional[str] = Header(default=None),
):
    """Return all active/suspended webhooks (secret redacted)."""
    _require_admin(authorization)
    from backend.services.webhook_manager import list_webhooks as _list
    return {"webhooks": _list()}


@router.delete("/{webhook_id}", summary="Delete webhook (admin only)")
async def delete_webhook(
    webhook_id: str,
    authorization: Optional[str] = Header(default=None),
):
    """Soft-delete a webhook by ID."""
    _require_admin(authorization)
    from backend.services.webhook_manager import delete_webhook as _delete
    ok = _delete(webhook_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Webhook {webhook_id!r} not found.")
    return {"deleted": True, "webhook_id": webhook_id}


@router.post("/{webhook_id}/test", summary="Send test delivery (admin only)")
async def send_test_delivery(
    webhook_id: str,
    authorization: Optional[str] = Header(default=None),
):
    """Queue a test delivery to the specified webhook."""
    _require_admin(authorization)
    from backend.services.webhook_manager import send_test as _test
    try:
        return _test(webhook_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/deliveries", summary="Delivery log (admin only)")
async def delivery_log(
    webhook_id: Optional[str] = None,
    limit: int = 50,
    authorization: Optional[str] = Header(default=None),
):
    """Return recent webhook delivery attempts, optionally filtered by webhook_id."""
    _require_admin(authorization)
    from backend.services.webhook_manager import get_deliveries as _deliveries
    return {"deliveries": _deliveries(webhook_id=webhook_id, limit=min(limit, 200))}
