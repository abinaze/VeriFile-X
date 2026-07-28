"""
Analyst feedback endpoint — Nash Equilibrium adaptive detection.

Routes:
  POST /api/v1/feedback          — submit correction (analyst/admin)
  GET  /api/v1/feedback          — list recent feedback (analyst/admin)
  GET  /api/v1/feedback/weights  — current adaptive weights (admin only)
"""
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

router = APIRouter(prefix="/api/v1/feedback", tags=["Feedback"])

# F-4: use the shared core.auth implementation instead of local copies.
from backend.core.auth import require_analyst as _require_analyst
from backend.core.auth import require_admin as _require_admin


class FeedbackRequest(BaseModel):
    evidence_id:      str
    true_label:       str                        # "ai_generated" | "authentic"
    predicted_label:  str
    signals:          List[Dict[str, Any]] = []
    analyst_notes:    Optional[str] = None


@router.post("/", summary="Submit analyst feedback (analyst/admin)")
async def submit_feedback(
    body: FeedbackRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """
    Record an analyst correction.

    When a result is wrong, submit the true label and the signal list
    from the forensic report.  Signal weights are updated automatically
    using the Nash gradient update rule.
    """
    _require_analyst(authorization)
    from backend.services.feedback_manager import record_feedback
    try:
        return record_feedback(
            evidence_id=body.evidence_id,
            true_label=body.true_label,
            predicted_label=body.predicted_label,
            signals=body.signals,
            analyst_notes=body.analyst_notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/", summary="List recent feedback (analyst/admin)")
async def list_feedback(
    evidence_id: Optional[str] = None,
    limit: int = 50,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Return recent feedback records, optionally filtered by evidence_id."""
    _require_analyst(authorization)
    from backend.services.feedback_manager import get_feedback_history
    return {
        "feedback": get_feedback_history(
            evidence_id=evidence_id,
            limit=min(limit, 200),
        )
    }


@router.get("/weights", summary="Current adaptive signal weights (admin)")
async def get_weights(
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Return current adaptive weight multipliers for all signals."""
    _require_admin(authorization)
    from backend.services.feedback_manager import get_weight_summary
    return get_weight_summary()
