"""
Append-only audit log for forensic analysis tracking.
Every analysis is recorded with timestamp, file hash, and verdict.

Rotation: file is moved to audit_log.jsonl.bak when it exceeds 50MB.
"""
import json
import threading
from backend.core.logger import setup_logger
import shutil
from datetime import datetime, timezone
from pathlib import Path

from backend.core.config import settings

logger = setup_logger(__name__)

AUDIT_LOG_PATH = Path(__file__).parent.parent / "data" / "audit_log.jsonl"
MAX_LOG_BYTES      = 50 * 1024 * 1024  # 50 MB
_audit_write_lock  = threading.Lock()


def log_analysis(
    evidence_id: str,
    filename: str,
    file_sha256: str,
    ai_probability: float,
    classification: str,
    total_signals: int,
    suspicious_signals: int,
    methods_used: list,
) -> dict:
    """Append one analysis record to the audit log. Returns the entry."""
    entry = {
        "evidence_id":    evidence_id,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "filename":       filename,
        "file_sha256":    file_sha256,
        "verdict": {
            "ai_probability":    round(ai_probability, 4),
            "classification":    classification,
            "total_signals":     total_signals,
            "suspicious_signals": suspicious_signals,
            "methods_used":      methods_used,
        },
        "analyzer_version": settings.VERSION,
    }

    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _audit_write_lock:
            # Rotate at 50MB — use timestamp to avoid clobbering previous backup
            if AUDIT_LOG_PATH.exists() and AUDIT_LOG_PATH.stat().st_size > MAX_LOG_BYTES:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                shutil.move(str(AUDIT_LOG_PATH), f"{AUDIT_LOG_PATH}.{ts}.bak")
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        logger.info("Audit log: recorded %s", evidence_id)
    except Exception:
        logger.error("Audit log write failed", exc_info=True)

    return entry


def get_recent_analyses(limit: int = 20) -> list:
    """Return the most recent N analysis records."""
    if not AUDIT_LOG_PATH.exists():
        return []
    try:
        lines = AUDIT_LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
        entries = [json.loads(ln) for ln in lines if ln.strip()]
        return list(reversed(entries))[:limit]
    except Exception:
        logger.error("Audit log read failed", exc_info=True)
        return []


def get_stats() -> dict:
    """Return aggregate stats from the audit log (last 1000 entries)."""
    entries = get_recent_analyses(limit=1000)
    if not entries:
        return {"total_analyses": 0}
    ai_count   = sum(1 for e in entries if "ai_generated" in e["verdict"]["classification"])
    real_count = sum(1 for e in entries if "authentic"    in e["verdict"]["classification"])
    return {
        "total_analyses":    len(entries),
        "ai_detected":       ai_count,
        "authentic_detected": real_count,
        "ambiguous":         len(entries) - ai_count - real_count,
    }
