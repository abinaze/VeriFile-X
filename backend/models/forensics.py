"""
Pydantic models for forensic analysis responses.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime


class FileInfo(BaseModel):
    """Basic file information."""
    filename: str
    format: Optional[str]
    mode: Optional[str]
    size: tuple
    width: int
    height: int
    file_size_bytes: int


class TamperingAnalysis(BaseModel):
    """Tampering detection results."""
    suspicious_flags: List[str]
    confidence: str
    analysis: List[str]


class ForensicReportSummary(BaseModel):
    """Summary of forensic analysis."""
    has_metadata: bool
    suspicious_flags_count: int
    authenticity_confidence: str


class ForensicReport(BaseModel):
    """Complete forensic analysis report."""
    metadata: Dict[str, Any]
    file_info: FileInfo
    exif_data: Dict[str, Any]
    hashes: Dict[str, str]
    tampering_analysis: TamperingAnalysis
    ai_detection: Dict[str, Any]
    summary: Dict[str, Any]


class AIDetectionSignals(BaseModel):
    """Detection signals for AI analysis."""
    noise: Dict[str, Any]
    frequency: Dict[str, Any]
    jpeg: Dict[str, Any]
    color: Dict[str, Any]


class AIDetection(BaseModel):
    """AI-generated image detection results."""
    ai_probability: float
    classification: str
    confidence: str
    detection_signals: AIDetectionSignals
    summary: Dict[str, int]
