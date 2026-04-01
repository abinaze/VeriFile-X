"""
Append-only audit log for forensic analysis tracking.
Every analysis is recorded with timestamp, file hash, and verdict.
This log is append-only — entries are never modified or deleted.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from backend.core.logger import setup_logger

logger = setup_logger(__name__)

AUDIT_LOG_PATH = Path("audit_log.jsonl")


def log_analysis(
    evidence_id: str,
    filename: str,
    file_sha256: str,
    ai_probability: float,
    classification: str,
    total_signals: int,
    suspicious_signals: int,
    methods_used: list
) -> dict:
    """
    Append one analysis record to the audit log.
    Returns the log entry dict.
    """
    entry = {
        "evidence_id": evidence_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "file_sha256": file_sha256,
        "verdict": {
            "ai_probability": round(ai_probability, 4),
            "classification": classification,
            "total_signals": total_signals,
            "suspicious_signals": suspicious_signals,
            "methods_used": methods_used
        },
        "analyzer_version": "6.0.0"
    }

    # Append to log file (never overwrite)
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Audit log: recorded {evidence_id}")
    except Exception as e:
        logger.error(f"Audit log write failed: {e}")

    return entry


def get_recent_analyses(limit: int = 20) -> list:
    """Return the most recent N analysis records."""
    if not AUDIT_LOG_PATH.exists():
        return []
    try:
        lines = AUDIT_LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
        entries = [json.loads(l) for l in lines if l.strip()]
        return list(reversed(entries))[:limit]
    except Exception as e:
        logger.error(f"Audit log read failed: {e}")
        return []


def get_stats() -> dict:
    """Return aggregate stats from the audit log."""
    entries = get_recent_analyses(limit=10000)
    if not entries:
        return {"total_analyses": 0}

    ai_count = sum(1 for e in entries if "ai_generated" in e["verdict"]["classification"])
    real_count = sum(1 for e in entries if "authentic" in e["verdict"]["classification"])

    return {
        "total_analyses": len(entries),
        "ai_detected": ai_count,
        "authentic_detected": real_count,
        "ambiguous": len(entries) - ai_count - real_count
    }
