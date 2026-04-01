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
    assert report["total_signals"] == 26
    assert len(report["all_signals"]) == 26
    
    # Check methods used
    assert "statistical" in report["methods_used"]
    assert "dire" in report["methods_used"]
    assert "clip" in report["methods_used"]
    assert "prnu" in report["methods_used"]
    
    # Check version
    assert report["detection_version"] == "advanced-ensemble-v1.4"
    
    # Cleanup
    detector.cleanup()


def test_advanced_ensemble_forensics_integration(sample_image_bytes):
    """Test integration with forensics service."""
    from backend.services.image_forensics import ImageForensics
    
    forensics = ImageForensics(sample_image_bytes, "test.png")
    report = forensics.generate_forensic_report()
    
    # Check advanced detection was used
    assert report["ai_detection"]["total_signals"] == 26
    assert report["metadata"]["analyzer_version"] == "6.0.0"
    assert "methods_used" in report["ai_detection"]
    assert len(report["ai_detection"]["methods_used"]) == 7
