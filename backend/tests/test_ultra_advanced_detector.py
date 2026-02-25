"""
Tests for ultra-advanced AI detection methods.
"""
import pytest
from backend.services.ultra_advanced_detector import UltraAdvancedDetector


def test_rgb_noise_covariance(sample_image_bytes):
    """Test RGB noise covariance analysis."""
    detector = UltraAdvancedDetector(sample_image_bytes, "test.png")
    result = detector.analyze_rgb_noise_covariance()
    
    assert result["signal_name"] == "RGB Noise Covariance"
    assert 0 <= result["score"] <= 1
    assert result["confidence"] == 0.88
    assert "raw_value" in result
    assert result["method"] == "cross_channel_noise_covariance"


def test_patch_spectral_variance(sample_image_bytes):
    """Test patch-level spectral variance analysis."""
    detector = UltraAdvancedDetector(sample_image_bytes, "test.png")
    result = detector.analyze_patch_spectral_variance()
    
    assert result["signal_name"] == "Patch Spectral Variance"
    assert 0 <= result["score"] <= 1
    assert "explanation" in result
    assert result["method"] == "patch_level_fft_variance"


def test_natural_prior_deviation(sample_image_bytes):
    """Test natural image prior deviation."""
    detector = UltraAdvancedDetector(sample_image_bytes, "test.png")
    result = detector.analyze_natural_prior_deviation()
    
    assert result["signal_name"] == "Natural Prior Deviation"
    assert 0 <= result["score"] <= 1
    assert result["confidence"] == 0.80
    assert result["method"] == "natural_image_prior"


def test_ultra_complete_detection(sample_image_bytes):
    """Test complete ultra-advanced detection workflow."""
    detector = UltraAdvancedDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    assert "ai_probability" in report
    assert "classification" in report
    assert "all_signals" in report
    assert len(report["all_signals"]) == 13  # 10 base + 3 new
    assert report["total_signals"] == 13
    assert report["detection_version"] == "ultra-advanced-v1.0"


def test_ultra_forensics_integration(sample_image_bytes):
    """Test integration with forensics service."""
    from backend.services.image_forensics import ImageForensics
    
    forensics = ImageForensics(sample_image_bytes, "test.png")
    report = forensics.generate_forensic_report()
    
    assert "ai_detection" in report
    assert report["ai_detection"]["total_signals"] == 13
    assert report["metadata"]["analyzer_version"] == "3.0.0"
    assert "detection_version" in report["ai_detection"]
