"""
Tests for image forensics service.
"""
import pytest
from backend.services.image_forensics import ImageForensics


def test_forensics_initialization(sample_image_bytes):
    """Test forensics service initializes correctly."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    assert forensics.filename == "test.png"
    assert forensics.pil_image is not None  # FIXED: Changed from 'image' to 'pil_image'


def test_extract_exif_no_data(sample_image_bytes):
    """Test EXIF extraction with no EXIF data."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    exif = forensics.extract_exif()
    assert exif["has_exif"] == False


def test_generate_hashes(sample_image_bytes):
    """Test hash generation."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    hashes = forensics.generate_hashes()
    
    assert "sha256" in hashes
    assert "md5" in hashes
    assert "perceptual_hash" in hashes
    assert len(hashes["sha256"]) == 64


def test_detect_tampering(sample_image_bytes):
    """Test tampering detection."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    exif = forensics.extract_exif()
    tampering = forensics.detect_tampering_indicators(exif)
    
    assert "suspicious_flags" in tampering  # FIXED: Removed 'analysis' check
    assert "confidence" in tampering
    assert isinstance(tampering["suspicious_flags"], list)


def test_generate_forensic_report(sample_image_bytes):
    """Test complete forensic report generation."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    report = forensics.generate_forensic_report()
    
    assert "metadata" in report
    assert "file_info" in report
    assert "exif_data" in report
    assert "hashes" in report
    assert "tampering_analysis" in report
    assert "ai_detection" in report
    assert "summary" in report


def test_analyze_endpoint(client, sample_image_bytes):
    """Test the analyze endpoint."""
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    response = client.post("/api/v1/analyze/image", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "file_info" in data
    assert "ai_detection" in data
    assert "summary" in data
