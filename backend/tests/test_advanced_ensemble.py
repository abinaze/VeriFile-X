"""
Tests for advanced ensemble detector.
"""
import pytest


def test_advanced_ensemble_initialization(sample_image_bytes):
    """Test advanced ensemble detector initialization."""
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    
    detector = AdvancedEnsembleDetector(sample_image_bytes, "test.png")
    assert detector is not None


def test_advanced_ensemble_complete_detection(sample_image_bytes):
    """Test complete advanced ensemble detection."""
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    
    detector = AdvancedEnsembleDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    # Check structure
    assert "ai_probability" in report
    assert "classification" in report
    assert "all_signals" in report
    assert "methods_used" in report
    
    # Should have 21 signals (19 statistical + DIRE + CLIP)
    assert report["total_signals"] == 30
    assert len(report["all_signals"]) == 30
    
    # Check methods used
    assert "statistical" in report["methods_used"]
    assert "dire" in report["methods_used"]
    assert "clip" in report["methods_used"]
    assert "prnu" in report["methods_used"]
    
    # Check version
    assert report["detection_version"] == "advanced-ensemble-v1.6"
    
    # Cleanup
    detector.cleanup()


def test_advanced_ensemble_forensics_integration(sample_image_bytes):
    """Test integration with forensics service."""
    from backend.services.image_forensics import ImageForensics
    
    forensics = ImageForensics(sample_image_bytes, "test.png")
    report = forensics.generate_forensic_report()
    
    # Check advanced detection was used
    assert report["ai_detection"]["total_signals"] == 30
    assert "analyzer_version" in report["metadata"]
    assert "methods_used" in report["ai_detection"]
    assert len(report["ai_detection"]["methods_used"]) == 12


class TestStatBundleConfidence:
    """F-13 regression tests: the statistical bundle's confidence must
    reflect how many of its 19 sub-signals actually succeeded, not be
    hardcoded to 1.0 regardless."""

    def test_all_high_confidence_signals_average_high(self):
        from backend.services.advanced_ensemble_detector import _aggregate_stat_confidence
        signals = [{"confidence": 0.92} for _ in range(19)]
        assert _aggregate_stat_confidence(signals) == pytest.approx(0.92)

    def test_some_failed_signals_drag_average_down(self):
        """This is the exact scenario F-13 fixes: several sub-signals
        hit their 'Analysis failed - insufficient data' fallback
        (confidence 0.3) -- the old hardcoded 1.0 ignored this entirely."""
        from backend.services.advanced_ensemble_detector import _aggregate_stat_confidence
        signals = [{"confidence": 0.9} for _ in range(10)] + [{"confidence": 0.3} for _ in range(9)]
        result = _aggregate_stat_confidence(signals)
        assert result < 0.9, "failed sub-signals should measurably reduce the aggregate"
        assert result == pytest.approx((0.9 * 10 + 0.3 * 9) / 19)

    def test_empty_signals_list_is_zero_confidence(self):
        from backend.services.advanced_ensemble_detector import _aggregate_stat_confidence
        assert _aggregate_stat_confidence([]) == 0.0

    def test_not_hardcoded_to_one(self):
        """Direct regression check against the exact old behavior."""
        from backend.services.advanced_ensemble_detector import _aggregate_stat_confidence
        signals = [{"confidence": 0.3} for _ in range(19)]  # worst case: all failed
        assert _aggregate_stat_confidence(signals) == pytest.approx(0.3)
        assert _aggregate_stat_confidence(signals) != 1.0
