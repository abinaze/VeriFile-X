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
import os
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


def _hash_key(raw_key: str, salt: str = "") -> str:
    """SHA-256 hash of a raw key, with an optional per-key salt (F-6).

    salt="" reproduces the exact legacy unsalted hash, so existing
    records created before this fix (which have no "salt" field) keep
    verifying correctly -- see verify_key() below.
    """
    return hashlib.sha256((salt + raw_key).encode("utf-8")).hexdigest()


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


def _rewrite_keys_file(keys: Dict[str, Dict[str, Any]]) -> None:
    """Atomically rewrite the whole keys file with exactly one line per
    key (F-10).

    Replaces the previous append-only-forever behavior: _save_key() used
    to append a full copy of an entry on every successful authentication
    (not just on creation), so data/api_keys.jsonl grew by one line per
    auth with no bound, and _load_keys() re-read and re-parsed every
    accumulated line on every single call. Writes to a temp file first
    and os.replace()'s it into place, which is atomic on both POSIX and
    Windows -- a crash mid-write can never leave a corrupt/partial file
    in KEYS_PATH.
    """
    KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = KEYS_PATH.with_suffix(KEYS_PATH.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in keys.values():
            f.write(json.dumps(entry) + "\n")
    os.replace(tmp_path, KEYS_PATH)


def _upsert_key(entry: Dict[str, Any]) -> None:
    """Insert or update a single key entry, compacted (F-10).

    Performs the full load-merge-rewrite sequence under one lock
    acquisition, so concurrent callers can no longer race on a stale
    read -- this is also what closes the use_count undercount the audit
    flagged alongside the plain unbounded-growth issue (previously only
    the final _save_key() append was lock-protected, not the read and
    increment before it).
    """
    with _key_write_lock:
        keys = _load_keys()
        keys[entry["key_id"]] = entry
        _rewrite_keys_file(keys)


def create_key(name: str, role: str = "analyst",
               description: str = "", created_by: str = "system") -> Dict[str, Any]:
    if role not in ROLES:
        return {"error": f"Invalid role. Must be one of: {list(ROLES.keys())}"}
    if not name or not name.strip():
        return {"error": "Key name is required."}

    raw_key  = f"vfx_{secrets.token_urlsafe(32)}"
    salt     = secrets.token_hex(16)
    key_hash = _hash_key(raw_key, salt)
    key_id   = str(uuid.uuid4())

    entry = {
        "key_id": key_id, "name": name.strip()[:100],
        "description": description.strip()[:500], "role": role,
        "key_hash": key_hash, "salt": salt, "created_at": _now(), "created_by": created_by,
        "last_used": None, "use_count": 0, "active": True,
    }
    _upsert_key(entry)
    logger.info(f"API key created: {key_id} name={name} role={role}")
    return {**entry, "key": raw_key,
            "warning": "Save this key now. It will not be shown again.",
            "key_hash": "[hidden]", "salt": "[hidden]"}


def verify_key(raw_key: str) -> Optional[Dict[str, Any]]:
    if not raw_key or not raw_key.startswith("vfx_"):
        return None
    keys = _load_keys()
    for key_id, entry in keys.items():
        # salt defaults to "" for legacy records created before F-6, which
        # reproduces the exact old unsalted hash -- so already-issued keys
        # keep verifying correctly with no migration step required.
        expected_hash = _hash_key(raw_key, entry.get("salt", ""))
        if (
            secrets.compare_digest(entry.get("key_hash", ""), expected_hash)
            and entry.get("active", False)
        ):
            # F-10: re-read the latest on-disk state and increment
            # use_count inside the SAME lock as the write. Previously
            # only the final _save_key() append was lock-protected, not
            # the read-and-increment before it -- two concurrent
            # requests using the same key could both read the same
            # stale use_count, and one increment would be lost.
            with _key_write_lock:
                latest_keys = _load_keys()
                latest_entry = latest_keys.get(key_id, entry)
                latest_entry["last_used"] = _now()
                latest_entry["use_count"] = latest_entry.get("use_count", 0) + 1
                latest_keys[key_id] = latest_entry
                _rewrite_keys_file(latest_keys)
            return {k: v for k, v in latest_entry.items() if k not in ("key_hash", "salt")}
    return None



def revoke_key(key_id: str, revoked_by: str = "system") -> Dict[str, Any]:
    with _key_write_lock:
        keys = _load_keys()
        if key_id not in keys:
            return {"error": f"Key not found: {key_id}"}
        entry = keys[key_id]
        entry["active"]     = False
        entry["revoked_at"] = _now()
        entry["revoked_by"] = revoked_by
        _rewrite_keys_file(keys)
    logger.info(f"API key revoked: {key_id}")
    return {k: v for k, v in entry.items() if k not in ("key_hash", "salt")}


def list_keys(include_inactive: bool = False) -> List[Dict[str, Any]]:
    keys   = _load_keys()
    result = []
    for entry in keys.values():
        if not include_inactive and not entry.get("active", False):
            continue
        result.append({k: v for k, v in entry.items() if k not in ("key_hash", "salt")})
    return sorted(result, key=lambda x: x.get("created_at", ""), reverse=True)
