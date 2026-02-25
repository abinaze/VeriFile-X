"""
Tests for advanced AI detection system.
"""
import pytest
from backend.services.advanced_ai_detector import AdvancedAIDetector


def test_advanced_detector_initialization(sample_image_bytes):
    """Test advanced detector initializes correctly."""
    detector = AdvancedAIDetector(sample_image_bytes, "test.png")
    assert detector.filename == "test.png"
    assert detector.cv_image is not None


def test_fft_radial_spectrum(sample_image_bytes):
    """Test FFT radial spectrum analysis."""
    detector = AdvancedAIDetector(sample_image_bytes, "test.png")
    result = detector.analyze_fft_radial_spectrum()
    
    assert "signal_name" in result
    assert "score" in result
    assert "confidence" in result
    assert "explanation" in result
    assert 0 <= result["score"] <= 1
    assert 0 <= result["confidence"] <= 1


def test_dct_coefficients(sample_image_bytes):
    """Test DCT coefficient analysis."""
    detector = AdvancedAIDetector(sample_image_bytes, "test.png")
    result = detector.analyze_dct_coefficients()
    
    assert result["signal_name"] == "DCT Coefficients"
    assert 0 <= result["score"] <= 1


def test_wavelet_energy(sample_image_bytes):
    """Test wavelet decomposition analysis."""
    detector = AdvancedAIDetector(sample_image_bytes, "test.png")
    result = detector.analyze_wavelet_energy()
    
    assert result["signal_name"] == "Wavelet Energy"
    assert "raw_value" in result


def test_complete_detection(sample_image_bytes):
    """Test complete advanced detection workflow."""
    detector = AdvancedAIDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    assert "ai_probability" in report
    assert "classification" in report
    assert "all_signals" in report
    assert "top_reasons" in report
    assert len(report["all_signals"]) == 10  # Base detector has 10 signals
    assert report["total_signals"] == 10


def test_forensics_integration(sample_image_bytes):
    """Test integration with forensics service."""
    from backend.services.image_forensics import ImageForensics
    
    forensics = ImageForensics(sample_image_bytes, "test.png")
    report = forensics.generate_forensic_report()
    
    assert "ai_detection" in report
    assert "all_signals" in report["ai_detection"]
    assert report["summary"]["total_detection_signals"] == 13
