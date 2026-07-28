"""
Evidence case management API endpoints.
"""
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.core.logger import setup_logger

logger = setup_logger(__name__)
# BUG FIX: previously no default_limits — settings.RATE_LIMIT_PER_MINUTE
# (declared in .env.example/render.yaml) had no effect anywhere. Wired
# here as the DEFAULT limit for any endpoint without its own explicit
# @limiter.limit(...) decorator — the 24 existing per-endpoint
# decorators are intentionally tuned differently per endpoint cost and
# are NOT touched by this change.
from backend.core.config import settings as _settings
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{_settings.RATE_LIMIT_PER_MINUTE}/minute"])

from backend.core.auth import require_role_for_method
from fastapi import Depends

router = APIRouter(
    prefix="/api/v1/cases",
    dependencies=[Depends(require_role_for_method)],
    tags=["Case Management"]
)


class CreateCaseRequest(BaseModel):
    name:        str           = Field(..., min_length=1, max_length=200)
    description: str           = Field("", max_length=2000)
    tags:        List[str]     = Field(default_factory=list)


class AddEvidenceRequest(BaseModel):
    evidence_id:    str   = Field(..., description="UUID from forensic report")
    filename:       str   = Field(..., description="Original filename")
    ai_probability: float = Field(..., ge=0.0, le=1.0)
    classification: str   = Field(...)
    notes:          str   = Field("", max_length=1000)
    tags:           List[str] = Field(default_factory=list)


class UpdateStatusRequest(BaseModel):
    status: str = Field(..., description="open | closed | archived")


@router.post("/", summary="Create investigation case")
@limiter.limit("20/minute")
async def create_case(request: Request, body: CreateCaseRequest):
    """Create a new named investigation case."""
    from backend.services.case_manager import create_case as _create
    result = _create(body.name, body.description, body.tags)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/", summary="List investigation cases")
@limiter.limit("30/minute")
async def list_cases(
    request: Request,
    status: Optional[str] = None,
    limit: int = 50,
):
    """List cases, optionally filtered by status (open|closed|archived)."""
    from backend.services.case_manager import list_cases as _list
    return {"cases": _list(status=status, limit=min(limit, 100))}


@router.get("/search", summary="Search cases")
@limiter.limit("20/minute")
async def search_cases(request: Request, q: str = "", limit: int = 20):
    """Full-text search across case name, description, tags, and evidence notes."""
    from backend.services.case_manager import search_cases as _search
    return {"cases": _search(q, limit=min(limit, 50))}


@router.get("/{case_id}", summary="Get case details")
@limiter.limit("30/minute")
async def get_case(request: Request, case_id: str):
    """Retrieve full case including all evidence items."""
    from backend.services.case_manager import get_case as _get
    result = _get(case_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{case_id}/summary", summary="Get case summary")
@limiter.limit("30/minute")
async def get_case_summary(request: Request, case_id: str):
    """Retrieve lightweight case summary with aggregate evidence statistics."""
    from backend.services.case_manager import get_case_summary as _summary
    result = _summary(case_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/{case_id}/evidence", summary="Add evidence to case")
@limiter.limit("20/minute")
async def add_evidence(request: Request, case_id: str, body: AddEvidenceRequest):
    """Attach a forensic analysis result to an existing case."""
    from backend.services.case_manager import add_evidence as _add
    result = _add(
        case_id=case_id,
        evidence_id=body.evidence_id,
        filename=body.filename,
        ai_probability=body.ai_probability,
        classification=body.classification,
        notes=body.notes,
        tags=body.tags,
    )
    if "error" in result:
        status = 404 if "not found" in result["error"].lower() else 400
        raise HTTPException(status_code=status, detail=result["error"])
    return result


@router.patch("/{case_id}/status", summary="Update case status")
@limiter.limit("20/minute")
async def update_status(request: Request, case_id: str, body: UpdateStatusRequest):
    """Change case status: open -> closed -> archived."""
    from backend.services.case_manager import update_status as _update
    result = _update(case_id, body.status)
    if "error" in result:
        status = 404 if "not found" in result["error"].lower() else 400
        raise HTTPException(status_code=status, detail=result["error"])
    return result


@router.delete("/{case_id}", summary="Archive (soft-delete) case")
@limiter.limit("10/minute")
async def delete_case(request: Request, case_id: str):
    """Soft-delete: sets status to archived. Cases are never physically removed."""
    from backend.services.case_manager import delete_case as _delete
    result = _delete(case_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return {"message": f"Case {case_id} archived.", "case": result}
