"""
Tests for CLIP reference database.
"""
import pytest
from pathlib import Path

@pytest.mark.slow
def test_clip_database_exists():
    """Test that CLIP database file exists."""
    database_path = Path("data/reference/clip_database.pkl")
    
    # Database should exist after running build_clip_database.py
    if database_path.exists():
        assert database_path.stat().st_size > 0
        print("✅ CLIP database found")
    else:
        pytest.skip("CLIP database not built yet. Run: python scripts/build_clip_database.py")

@pytest.mark.slow
def test_clip_detector_loads_database():
    """Test that CLIP detector loads reference database."""
    from backend.services.clip_detector import CLIPDetector
    
    detector = CLIPDetector()
    detector._load_model()
    
    # Should have centroids loaded
    assert detector.real_centroid is not None
    assert detector.fake_centroid is not None
    
    # Centroids should be normalized
    real_norm = detector.real_centroid.norm().item()
    fake_norm = detector.fake_centroid.norm().item()
    
    assert 0.99 < real_norm < 1.01, f"Real centroid not normalized: {real_norm}"
    assert 0.99 < fake_norm < 1.01, f"Fake centroid not normalized: {fake_norm}"
    
    detector.cleanup()

@pytest.mark.slow
def test_clip_detection_with_database(sample_image_bytes):
    """Test CLIP detection uses database."""
    from backend.services.clip_detector import CLIPDetector
    
    detector = CLIPDetector()
    result = detector.detect(sample_image_bytes, "test.png")
    
    # Should return valid result
    assert 0 <= result["score"] <= 1
    assert result["confidence"] > 0
    
    detector.cleanup()
