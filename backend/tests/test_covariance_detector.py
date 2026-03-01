"""
Tests for covariance and eigenvalue detection methods.
"""
import pytest
from backend.services.covariance_detector import CovarianceDetector


def test_eigenvalue_spread(sample_image_bytes):
    """Test eigenvalue spread analysis."""
    detector = CovarianceDetector(sample_image_bytes, "test.png")
    result = detector.analyze_eigenvalue_spread()
    
    assert result["signal_name"] == "Eigenvalue Spread"
    assert 0 <= result["score"] <= 1
    assert result["confidence"] == 0.90
    assert "raw_value" in result
    assert result["method"] == "covariance_eigenvalue_analysis"
    assert result["raw_value"] > 0  # Should have positive spread


def test_local_covariance_consistency(sample_image_bytes):
    """Test local covariance consistency analysis."""
    detector = CovarianceDetector(sample_image_bytes, "test.png")
    result = detector.analyze_local_covariance_consistency()
    
    assert result["signal_name"] == "Local Covariance Consistency"
    assert 0 <= result["score"] <= 1
    assert result["confidence"] >= 0.3  # May be low for small images
    assert result["method"] == "local_covariance_consistency"


def test_patch_anisotropy_variance(sample_image_bytes):
    """Test patch anisotropy variance analysis."""
    detector = CovarianceDetector(sample_image_bytes, "test.png")
    result = detector.analyze_patch_anisotropy_variance()
    
    assert result["signal_name"] == "Patch Anisotropy Variance"
    assert 0 <= result["score"] <= 1
    assert result["confidence"] >= 0.3  # May be low for small images
    assert result["method"] == "patch_anisotropy_variance"


def test_covariance_complete_detection(sample_image_bytes):
    """Test complete covariance detection workflow."""
    detector = CovarianceDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    assert "ai_probability" in report
    assert "classification" in report
    assert "all_signals" in report
    assert len(report["all_signals"]) == 16  # 13 base + 3 new
    assert report["total_signals"] == 16
    assert report["detection_version"] == "covariance-advanced-v1.0"


def test_covariance_forensics_integration(sample_image_bytes):
    """Test integration with forensics service."""
    from backend.services.image_forensics import ImageForensics
    
    forensics = ImageForensics(sample_image_bytes, "test.png")
    report = forensics.generate_forensic_report()
    
    assert "ai_detection" in report
    assert report["ai_detection"]["total_signals"] == 19
    assert report["metadata"]["analyzer_version"] == "5.0.0"
    assert "detection_version" in report["ai_detection"]


def test_eigenvalue_spread_mathematical_properties(sample_image_bytes):
    """Test mathematical properties of eigenvalue analysis."""
    detector = CovarianceDetector(sample_image_bytes, "test.png")
    result = detector.analyze_eigenvalue_spread()
    
    # Spread should be positive
    assert result["raw_value"] > 0
    
    # Confidence should be high for this method
    assert result["confidence"] == 0.90
    
    # Expected range should be documented
    assert "expected_range" in result


def test_covariance_signal_ordering(sample_image_bytes):
    """Test that signals are properly ordered by suspicion level."""
    detector = CovarianceDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    # Top reasons should be from most suspicious signals
    all_scores = [s["score"] for s in report["all_signals"]]
    max_score = max(all_scores)
    
    # At least one top reason should be from high-scoring signal
    assert any(
        signal["explanation"] in report["top_reasons"]
        for signal in report["all_signals"]
        if signal["score"] >= max_score * 0.8
    )
