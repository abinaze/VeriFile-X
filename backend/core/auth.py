"""
Shared FastAPI auth dependencies.

Previously, _require_admin/_require_analyst were copy-pasted verbatim
across feedback.py, keys.py, and webhooks.py — and never added at all
to analyze.py/cases.py, leaving the entire analysis and case-management
API surface open with no authentication.

Centralizing here means: (1) one implementation to fix/audit instead of
four, (2) it's now trivial to apply to any router via
`dependencies=[Depends(require_analyst)]` at router-construction time.
"""
from typing import Optional
from fastapi import Header, HTTPException


def require_analyst(authorization: Optional[str] = Header(None)) -> dict:
    """FastAPI dependency: require a valid analyst- or admin-role API key."""
    from backend.services.api_key_manager import verify_key
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required.")
    entry = verify_key(authorization.removeprefix("Bearer ").strip())
    if not entry:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key.")
    if entry.get("role") not in ("admin", "analyst"):
        raise HTTPException(status_code=403, detail="Analyst or admin role required.")
    return entry


def require_admin(authorization: Optional[str] = Header(None)) -> dict:
    """FastAPI dependency: require a valid admin-role API key."""
    from backend.services.api_key_manager import verify_key
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required.")
    entry = verify_key(authorization.removeprefix("Bearer ").strip())
    if not entry:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key.")
    if entry.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return entry


def require_analyst_or_demo(authorization: Optional[str] = Header(None)) -> dict:
    """FastAPI dependency: require a real analyst/admin key, OR the public
    demo token (F-1). The demo token is opt-in and fail-closed: it is only
    accepted when settings.PUBLIC_DEMO_KEY is non-empty, so a deployment
    that never sets it behaves exactly like require_analyst. Still subject
    to normal per-IP rate limiting like any other request.
    """
    from backend.core.config import settings
    if (
        settings.PUBLIC_DEMO_KEY
        and authorization == f"Bearer {settings.PUBLIC_DEMO_KEY}"
    ):
        return {"key_id": "public-demo", "name": "Public Demo", "role": "analyst"}
    return require_analyst(authorization)
