"""
Evidence Case Management System.

Organises forensic analysis results into named investigation cases.
Cases are persisted to an append-only JSONL file (same pattern as audit_log).

A case contains:
  - case_id      UUID, immutable
  - name         Human-readable label
  - description  Optional free-text
  - created_at   ISO timestamp
  - status       open | closed | archived
  - evidence     List of evidence_id references with metadata
  - tags         Free-form list of strings

Operations:
  create_case   Create new case, return case_id
  add_evidence  Attach an evidence_id (from forensic report) to a case
  get_case      Retrieve full case with all evidence summaries
  list_cases    List all cases with status filter
  update_status Change case status
  search_cases  Full-text search across name, description, tags

Storage:
  cases.jsonl   — append-only case log (never overwritten)
  Each line is one full case snapshot. Last snapshot for each case_id wins.
"""
import json
import uuid
from backend.core.logger import setup_logger
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = setup_logger(__name__)

CASES_PATH = Path(__file__).parent.parent / "data" / "cases.jsonl"
_VALID_STATUSES = {"open", "closed", "archived"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_all_cases() -> Dict[str, Dict[str, Any]]:
    """Load all cases from JSONL. Last snapshot per case_id wins."""
    cases: Dict[str, Dict[str, Any]] = {}
    if not CASES_PATH.exists():
        return cases
    try:
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                cid  = case.get("case_id")
                if cid:
                    cases[cid] = case
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logger.error(f"Failed to load cases: {e}")
    return cases


def _save_case(case: Dict[str, Any]) -> None:
    """Append case snapshot to JSONL file."""
    try:
        with open(CASES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(case) + "\n")
    except Exception as e:
        logger.error(f"Failed to save case: {e}")


def create_case(
    name: str,
    description: str = "",
    tags: List[str] = None,
) -> Dict[str, Any]:
    """
    Create a new investigation case.

    Returns the full case dict including generated case_id.
    """
    if not name or not name.strip():
        return {"error": "Case name is required."}

    case = {
        "case_id":     str(uuid.uuid4()),
        "name":        name.strip()[:200],
        "description": description.strip()[:2000],
        "tags":        [t.strip() for t in (tags or []) if t.strip()][:20],
        "status":      "open",
        "created_at":  _now(),
        "updated_at":  _now(),
        "evidence":    [],
    }
    _save_case(case)
    logger.info(f"Case created: {case['case_id']} — {case['name']}")
    return case


def add_evidence(
    case_id: str,
    evidence_id: str,
    filename: str,
    ai_probability: float,
    classification: str,
    notes: str = "",
    tags: List[str] = None,
) -> Dict[str, Any]:
    """
    Add an evidence item to an existing case.

    Args:
        case_id:        Target case UUID
        evidence_id:    UUID from forensic report
        filename:       Original filename
        ai_probability: AI detection score 0.0-1.0
        classification: e.g. likely_ai_generated
        notes:          Free-text investigator notes
        tags:           Optional evidence-level tags

    Returns:
        Updated case dict, or error dict.
    """
    cases = _load_all_cases()
    if case_id not in cases:
        return {"error": f"Case not found: {case_id}"}

    case = cases[case_id]
    if case["status"] == "closed":
        return {"error": "Cannot add evidence to a closed case."}

    entry = {
        "evidence_id":    evidence_id,
        "filename":       filename,
        "ai_probability": round(float(ai_probability), 4),
        "classification": classification,
        "notes":          notes.strip()[:1000],
        "tags":           [t.strip() for t in (tags or []) if t.strip()][:10],
        "added_at":       _now(),
    }

    case["evidence"].append(entry)
    case["updated_at"] = _now()
    _save_case(case)

    logger.info(
        f"Evidence added to case {case_id}: "
        f"{evidence_id} ({classification}, p={ai_probability:.3f})"
    )
    return case


def get_case(case_id: str) -> Dict[str, Any]:
    """Retrieve a case by ID."""
    cases = _load_all_cases()
    if case_id not in cases:
        return {"error": f"Case not found: {case_id}"}
    return cases[case_id]


def list_cases(
    status: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    List cases, optionally filtered by status.

    Returns cases sorted by updated_at descending.
    """
    cases = _load_all_cases()
    result = list(cases.values())

    if status and status in _VALID_STATUSES:
        result = [c for c in result if c["status"] == status]

    result.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return result[:limit]


def update_status(case_id: str, new_status: str) -> Dict[str, Any]:
    """Update case status. Valid: open, closed, archived."""
    if new_status not in _VALID_STATUSES:
        return {"error": f"Invalid status. Must be one of: {_VALID_STATUSES}"}

    cases = _load_all_cases()
    if case_id not in cases:
        return {"error": f"Case not found: {case_id}"}

    case = cases[case_id]
    case["status"]     = new_status
    case["updated_at"] = _now()
    _save_case(case)

    logger.info(f"Case {case_id} status changed to {new_status}")
    return case


def search_cases(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Full-text search across case name, description, and tags.

    Case-insensitive substring match.
    """
    if not query or not query.strip():
        return list_cases(limit=limit)

    q      = query.strip().lower()
    cases  = _load_all_cases()
    result = []

    for case in cases.values():
        searchable = " ".join([
            case.get("name", ""),
            case.get("description", ""),
            " ".join(case.get("tags", [])),
            " ".join(
                e.get("filename", "") + " " + e.get("notes", "")
                for e in case.get("evidence", [])
            ),
        ]).lower()

        if q in searchable:
            result.append(case)

    result.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return result[:limit]


def delete_case(case_id: str) -> Dict[str, Any]:
    """
    Soft-delete: sets status to archived and saves tombstone.
    Cases are never physically removed (append-only log).
    """
    return update_status(case_id, "archived")


def get_case_summary(case_id: str) -> Dict[str, Any]:
    """Return lightweight case summary with aggregate evidence stats."""
    case = get_case(case_id)
    if "error" in case:
        return case

    evidence     = case.get("evidence", [])
    ai_probs     = [e["ai_probability"] for e in evidence]
    ai_detected  = sum(1 for e in evidence if "ai_generated" in e.get("classification", ""))

    return {
        "case_id":          case["case_id"],
        "name":             case["name"],
        "status":           case["status"],
        "created_at":       case["created_at"],
        "updated_at":       case["updated_at"],
        "evidence_count":   len(evidence),
        "ai_detected":      ai_detected,
        "mean_ai_prob":     round(sum(ai_probs) / len(ai_probs), 4) if ai_probs else 0.0,
        "tags":             case.get("tags", []),
    }
