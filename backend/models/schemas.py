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
    extension_valid: bool = True  # now actually enforced in validators.py
    size_bytes: int
    size_mb: float
    filename: str
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = None
