"""
Request/Response schemas for API endpoints.
Why: Type validation, auto-documentation, IDE support.
"""
from pydantic import BaseModel, Field
from typing import Optional


class FileValidationResponse(BaseModel):
    """Response model for file validation."""
    valid: bool
    mime_type: str
    extension: str
    extension_valid: bool = True  # BUG FIX: field didn't exist — Pydantic
    # v2's extra="ignore" default silently dropped it from validate_file()'s
    # result dict, so /api/v1/upload/validate could never actually surface
    # an extension/MIME mismatch to API consumers even before the
    # enforcement fix in validators.py.
    size_bytes: int
    size_mb: float
    filename: str
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = None
