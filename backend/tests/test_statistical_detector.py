"""
Tests for statistical modeling detection methods.
"""
import pytest
from backend.services.statistical_detector import StatisticalDetector


def test_mahalanobis_distance(sample_image_bytes):
    """Test Mahalanobis distance analysis."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_mahalanobis_distance()
    
    assert result["signal_name"] == "Mahalanobis Distance"
    assert 0 <= result["score"] <= 1
    assert result["confidence"] >= 0.3  # May be low on failure
    assert "raw_value" in result
    assert result["method"] == "mahalanobis_frequency_distance"


def test_kl_divergence(sample_image_bytes):
    """Test KL divergence analysis."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_kl_divergence()
    
    assert result["signal_name"] == "KL Divergence"
    assert 0 <= result["score"] <= 1
    assert result["confidence"] >= 0.3
    assert result["method"] == "kl_divergence_natural_prior"


def test_perturbation_stability(sample_image_bytes):
    """Test perturbation stability analysis."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_perturbation_stability()
    
    assert result["signal_name"] == "Perturbation Stability"
    assert 0 <= result["score"] <= 1
    assert result["confidence"] >= 0.3
    assert result["method"] == "perturbation_stability_test"


def test_statistical_complete_detection(sample_image_bytes):
    """Test complete statistical detection workflow."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    assert "ai_probability" in report
    assert "classification" in report
    assert "all_signals" in report
    assert len(report["all_signals"]) == 19  # 16 base + 3 statistical
    assert report["total_signals"] == 21
    assert report["detection_version"] == "statistical-modeling-v1.0"


def test_statistical_forensics_integration(sample_image_bytes):
    """Test integration with forensics service."""
    from backend.services.image_forensics import ImageForensics
    
    forensics = ImageForensics(sample_image_bytes, "test.png")
    report = forensics.generate_forensic_report()
    
    assert "ai_detection" in report
    assert report["ai_detection"]["total_signals"] == 21
    assert report["metadata"]["analyzer_version"] == "6.0.0"
    assert "detection_version" in report["ai_detection"]


def test_mahalanobis_handles_errors_gracefully(sample_image_bytes):
    """Test that Mahalanobis analysis handles errors gracefully."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_mahalanobis_distance()
    
    # Should always return valid structure even on error
    assert "signal_name" in result
    assert "score" in result
    assert "confidence" in result
    assert isinstance(result["score"], float)


def test_statistical_signal_ordering(sample_image_bytes):
    """Test that signals are properly ordered by suspicion level."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
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


def test_statistical_confidence_weighting(sample_image_bytes):
    """Test that confidence weighting is applied correctly."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    # Verify weighted scoring
    total_weight = sum(s["confidence"] for s in report["all_signals"])
    expected_score = sum(
        s["score"] * s["confidence"] 
        for s in report["all_signals"]
    ) / total_weight
    
    # Score should be close to expected (within boost range)
    assert 0 <= report["ai_probability"] <= 1.0
    assert abs(report["ai_probability"] - expected_score) < 0.5  # Allow for boost
