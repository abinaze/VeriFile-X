"""API Key Management endpoints — admin only."""
from fastapi import APIRouter, HTTPException, Request, Header
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, Field
from typing import Optional
from backend.core.logger import setup_logger
from backend.core.auth import require_admin as _require_admin

logger = setup_logger(__name__)
# Wires RATE_LIMIT_PER_MINUTE as the default limit for any endpoint
# without its own explicit @limiter.limit(...) decorator.
from backend.core.config import settings as _settings
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{_settings.RATE_LIMIT_PER_MINUTE}/minute"])
router  = APIRouter(prefix="/api/v1/keys", tags=["API Key Management"])


class CreateKeyRequest(BaseModel):
    name:        str = Field(..., min_length=1, max_length=100)
    role:        str = Field("analyst")
    description: str = Field("", max_length=500)


@router.post("/", summary="Create API key (admin only)")
@limiter.limit("10/minute")
async def create_key(request: Request, body: CreateKeyRequest,
                     authorization: Optional[str] = Header(None)):
    _require_admin(authorization)
    from backend.services.api_key_manager import create_key as _create
    result = _create(body.name, body.role, body.description)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/", summary="List API keys (admin only)")
@limiter.limit("20/minute")
async def list_keys(request: Request, include_inactive: bool = False,
                    authorization: Optional[str] = Header(None)):
    _require_admin(authorization)
    from backend.services.api_key_manager import list_keys as _list
    return {"keys": _list(include_inactive=include_inactive)}


@router.delete("/{key_id}", summary="Revoke API key (admin only)")
@limiter.limit("10/minute")
async def revoke_key(request: Request, key_id: str,
                     authorization: Optional[str] = Header(None)):
    admin = _require_admin(authorization)
    from backend.services.api_key_manager import revoke_key as _revoke
    result = _revoke(key_id, revoked_by=admin.get("name", "admin"))
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return {"message": f"Key {key_id} revoked.", "key": result}


@router.get("/verify", summary="Verify own API key")
@limiter.limit("30/minute")
async def verify_key_endpoint(request: Request,
                               authorization: Optional[str] = Header(None)):
    from backend.services.api_key_manager import verify_key
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header required.")
    entry = verify_key(authorization.removeprefix("Bearer ").strip())
    if not entry:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key.")
    return {"valid": True, "key_id": entry["key_id"], "name": entry["name"],
            "role": entry["role"], "use_count": entry["use_count"],
            "last_used": entry["last_used"]}
