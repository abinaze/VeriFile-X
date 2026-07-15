"""
API Key Management and RBAC.
Raw keys are never stored — only SHA-256 hash is persisted.
Storage: api_keys.jsonl (append-only)

Roles:
  viewer  - GET only
  analyst - GET + POST (all analysis endpoints)
  admin   - all methods including key management
"""
import json
import uuid
import hashlib
import secrets
from backend.core.logger import setup_logger
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = setup_logger(__name__)

import threading as _threading
_key_write_lock = _threading.Lock()
KEYS_PATH = Path(__file__).parent.parent / "data" / "api_keys.jsonl"

ROLES = {
    "viewer":  {"GET"},
    "analyst": {"GET", "POST"},
    "admin":   {"GET", "POST", "PATCH", "DELETE"},
}



def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _load_keys() -> Dict[str, Dict[str, Any]]:
    keys: Dict[str, Dict[str, Any]] = {}
    if not KEYS_PATH.exists():
        return keys
    try:
        for line in KEYS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                kid   = entry.get("key_id")
                if kid:
                    keys[kid] = entry
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logger.error(f"Failed to load API keys: {e}")
    return keys


def _save_key(entry: Dict[str, Any]) -> None:
    try:
        with open(KEYS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to save API key: {e}")


def create_key(name: str, role: str = "analyst",
               description: str = "", created_by: str = "system") -> Dict[str, Any]:
    if role not in ROLES:
        return {"error": f"Invalid role. Must be one of: {list(ROLES.keys())}"}
    if not name or not name.strip():
        return {"error": "Key name is required."}

    raw_key  = f"vfx_{secrets.token_urlsafe(32)}"
    key_hash = _hash_key(raw_key)
    key_id   = str(uuid.uuid4())

    entry = {
        "key_id": key_id, "name": name.strip()[:100],
        "description": description.strip()[:500], "role": role,
        "key_hash": key_hash, "created_at": _now(), "created_by": created_by,
        "last_used": None, "use_count": 0, "active": True,
    }
    _save_key(entry)
    logger.info(f"API key created: {key_id} name={name} role={role}")
    return {**entry, "key": raw_key,
            "warning": "Save this key now. It will not be shown again.",
            "key_hash": "[hidden]"}


def verify_key(raw_key: str) -> Optional[Dict[str, Any]]:
    if not raw_key or not raw_key.startswith("vfx_"):
        return None
    key_hash = _hash_key(raw_key)
    keys     = _load_keys()
    for entry in keys.values():
        if entry.get("key_hash") == key_hash and entry.get("active", False):
            entry["last_used"] = _now()
            entry["use_count"] = entry.get("use_count", 0) + 1
            with _key_write_lock:
                _save_key(entry)
            return {k: v for k, v in entry.items() if k != "key_hash"}
    return None



def revoke_key(key_id: str, revoked_by: str = "system") -> Dict[str, Any]:
    keys = _load_keys()
    if key_id not in keys:
        return {"error": f"Key not found: {key_id}"}
    entry = keys[key_id]
    entry["active"]     = False
    entry["revoked_at"] = _now()
    entry["revoked_by"] = revoked_by
    _save_key(entry)
    logger.info(f"API key revoked: {key_id}")
    return {k: v for k, v in entry.items() if k != "key_hash"}


def list_keys(include_inactive: bool = False) -> List[Dict[str, Any]]:
    keys   = _load_keys()
    result = []
    for entry in keys.values():
        if not include_inactive and not entry.get("active", False):
            continue
        result.append({k: v for k, v in entry.items() if k != "key_hash"})
    return sorted(result, key=lambda x: x.get("created_at", ""), reverse=True)
