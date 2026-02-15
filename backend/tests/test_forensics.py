"""
Tests for image forensics analysis.
"""
import pytest
from backend.services.image_forensics import ImageForensics


def test_forensics_initialization(sample_image_bytes):
    """Test forensics analyzer initializes correctly."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    assert forensics.filename == "test.png"
    assert forensics.image is not None


def test_extract_exif_no_data(sample_image_bytes):
    """Test EXIF extraction on image without EXIF."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    exif = forensics.extract_exif()
    
    # Test PNG has no EXIF
    assert exif["has_exif"] == False


def test_generate_hashes(sample_image_bytes):
    """Test hash generation."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    hashes = forensics.generate_hashes()
    
    assert "sha256" in hashes
    assert "md5" in hashes
    assert "perceptual_hash" in hashes
    assert len(hashes["sha256"]) == 64  # SHA-256 is 64 hex chars
    assert len(hashes["md5"]) == 32      # MD5 is 32 hex chars


def test_detect_tampering(sample_image_bytes):
    """Test tampering detection."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    exif = forensics.extract_exif()
    tampering = forensics.detect_tampering_indicators(exif)
    
    assert "suspicious_flags" in tampering
    assert "confidence" in tampering
    assert "analysis" in tampering


def test_generate_forensic_report(sample_image_bytes):
    """Test complete forensic report generation."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    report = forensics.generate_forensic_report()
    
    assert "metadata" in report
    assert "file_info" in report
    assert "exif_data" in report
    assert "hashes" in report
    assert "tampering_analysis" in report
    assert "summary" in report
    
    # Verify summary structure
    assert "has_metadata" in report["summary"]
    assert "suspicious_flags_count" in report["summary"]
    assert "authenticity_confidence" in report["summary"]


def test_analyze_endpoint(client, sample_image_bytes):
    """Test forensic analysis endpoint."""
    from io import BytesIO
    
    files = {"file": ("test.png", BytesIO(sample_image_bytes), "image/png")}
    response = client.post("/api/v1/analyze/image", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "file_info" in data
    assert "hashes" in data
    assert data["file_info"]["filename"] == "test.png"
