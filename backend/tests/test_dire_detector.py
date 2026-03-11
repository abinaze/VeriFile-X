"""
Tests for DIRE detector.
"""
import pytest
from PIL import Image
import io

@pytest.mark.slow
def test_dire_detector_initialization():
    """Test DIRE detector initializes correctly."""
    from backend.services.dire_detector import DIREDetector
    
    detector = DIREDetector()
    
    # Should initialize without loading model
    assert detector.device in ["cpu", "cuda"]
    assert detector._model_loaded == False
    assert detector.cache_key == "stable-diffusion-2-1"

@pytest.mark.slow
def test_dire_detection_on_sample(sample_image_bytes):
    """Test DIRE detection on sample image."""
    from backend.services.dire_detector import DIREDetector
    
    detector = DIREDetector()
    result = detector.detect(sample_image_bytes, "test.png")
    
    # Should return valid result structure
    assert "signal_name" in result
    assert "score" in result
    assert "confidence" in result
    assert result["method"] == "diffusion_reconstruction"  # Fixed!
    assert 0 <= result["score"] <= 1
    
    detector.cleanup()

@pytest.mark.slow
def test_dire_handles_errors_gracefully():
    """Test DIRE handles invalid input gracefully."""
    from backend.services.dire_detector import DIREDetector
    
    detector = DIREDetector()
    
    # Invalid image bytes
    result = detector.detect(b"not an image", "invalid.png")
    
    # Should return neutral result on error
    assert result["score"] == 0.5
    assert result["confidence"] < 0.5
    
    detector.cleanup()
