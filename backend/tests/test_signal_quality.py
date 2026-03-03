"""
Tests for signal quality and mathematical properties.
"""
import pytest
import numpy as np  # ADD THIS IMPORT
from backend.services.statistical_detector import StatisticalDetector


def test_fft_radial_output_range(sample_image_bytes):
    """Validate FFT radial spectrum output properties."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_fft_radial_spectrum()
    
    assert isinstance(result["raw_value"], float)
    assert result["raw_value"] >= 0
    assert result["score"] >= 0 and result["score"] <= 1
    assert result["confidence"] > 0 and result["confidence"] <= 1


def test_eigenvalue_spread_bounds(sample_image_bytes):
    """Test eigenvalue spread mathematical constraints."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_eigenvalue_spread()
    
    assert result["raw_value"] > 0
    assert not np.isinf(result["raw_value"])
    assert not np.isnan(result["raw_value"])


def test_mahalanobis_distance_validity(sample_image_bytes):
    """Test Mahalanobis distance mathematical validity."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_mahalanobis_distance()
    
    assert result["raw_value"] >= 0
    assert np.isfinite(result["raw_value"])


def test_kl_divergence_properties(sample_image_bytes):
    """Test KL divergence non-negativity."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_kl_divergence()
    
    assert result["raw_value"] >= 0


def test_all_signals_have_required_fields(sample_image_bytes):
    """Ensure all signals return required fields."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    required_fields = ["signal_name", "score", "confidence", "explanation", "raw_value", "expected_range"]
    
    for signal in report["all_signals"]:
        for field in required_fields:
            assert field in signal, f"Signal {signal.get('signal_name', 'unknown')} missing field: {field}"


def test_signal_scores_are_bounded(sample_image_bytes):
    """Test that all signal scores are in [0, 1]."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    for signal in report["all_signals"]:
        assert 0 <= signal["score"] <= 1, f"Signal {signal['signal_name']} score out of bounds: {signal['score']}"


def test_confidence_values_are_valid(sample_image_bytes):
    """Test that all confidence values are in (0, 1]."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    for signal in report["all_signals"]:
        assert 0 < signal["confidence"] <= 1, f"Signal {signal['signal_name']} confidence invalid: {signal['confidence']}"


def test_weighted_score_calculation(sample_image_bytes):
    """Verify weighted score is calculated correctly."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    total_weight = sum(s["confidence"] for s in report["all_signals"])
    expected_base_score = sum(s["score"] * s["confidence"] for s in report["all_signals"]) / total_weight
    
    assert abs(report["ai_probability"] - expected_base_score) < 0.5


def test_suspicious_count_accuracy(sample_image_bytes):
    """Verify suspicious signal count is accurate."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    actual_suspicious = sum(1 for s in report["all_signals"] if s["score"] > 0.5)
    
    assert report["suspicious_signals_count"] == actual_suspicious


def test_radial_slope_in_expected_range(sample_image_bytes):
    """Test FFT radial slope is in natural range."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_fft_radial_spectrum()
    
    assert 0.1 < result["raw_value"] < 5.0, f"Radial slope {result['raw_value']} seems unrealistic"
