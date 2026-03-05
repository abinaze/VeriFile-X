"""
Tests for DIRE detector.
"""
import pytest


def test_dire_detector_initialization():
    """Test DIRE detector can be initialized."""
    from backend.services.dire_detector import DIREDetector
    
    detector = DIREDetector()
    assert detector is not None
    assert detector.device in ["cuda", "cpu"]


def test_dire_detection_on_sample(sample_image_bytes):
    """Test DIRE detection on sample image."""
    from backend.services.dire_detector import DIREDetector
    
    detector = DIREDetector()
    result = detector.detect(sample_image_bytes, "test.png")
    
    # Check structure
    assert "signal_name" in result
    assert result["signal_name"] == "DIRE Reconstruction Error"
    assert "score" in result
    assert "confidence" in result
    assert "explanation" in result
    assert "method" in result
    
    # Check values
    assert 0 <= result["score"] <= 1
    assert result["method"] == "diffusion_reconstruction_error"
    
    # Cleanup
    detector.cleanup()


def test_dire_handles_errors_gracefully():
    """Test DIRE handles corrupted input gracefully."""
    from backend.services.dire_detector import DIREDetector
    
    detector = DIREDetector()
    
    # Should not crash on bad input
    result = detector.detect(b"not an image", "bad.png")
    
    # Should return neutral score with low confidence
    assert result["score"] == 0.5
    assert result["confidence"] == 0.1
    
    detector.cleanup()
