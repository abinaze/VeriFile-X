"""
Tests for CLIP detector.
"""
import pytest

@pytest.mark.slow
def test_clip_detector_initialization():
    """Test CLIP detector can be initialized."""
    from backend.services.clip_detector import CLIPDetector
    
    detector = CLIPDetector()
    assert detector is not None
    assert detector.device in ["cuda", "cpu"]

@pytest.mark.slow
def test_clip_detection_on_sample(sample_image_bytes):
    """Test CLIP detection on sample image."""
    from backend.services.clip_detector import CLIPDetector
    
    detector = CLIPDetector()
    result = detector.detect(sample_image_bytes, "test.png")
    
    # Check structure
    assert "signal_name" in result
    assert result["signal_name"] == "CLIP Embedding Analysis"
    assert "score" in result
    assert "confidence" in result
    assert "explanation" in result
    assert "method" in result
    
    # Check values
    assert 0 <= result["score"] <= 1
    assert result["method"] == "clip_embedding_similarity"
    
    # Cleanup
    detector.cleanup()

@pytest.mark.slow
def test_clip_handles_errors_gracefully():
    """Test CLIP handles corrupted input gracefully."""
    from backend.services.clip_detector import CLIPDetector
    
    detector = CLIPDetector()
    
    # Should not crash on bad input
    result = detector.detect(b"not an image", "bad.png")
    
    # Should return neutral score with low confidence
    assert result["score"] == 0.5
    assert result["confidence"] == 0.1
    
    detector.cleanup()
