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


def test_load_model_is_not_duplicated_under_concurrency(monkeypatch):
    """F-7 regression test: concurrent cold-start requests for the same
    cache key must not each independently load the full ~4-5GB Stable
    Diffusion 2.1 pipeline. Same shape as clip_detector.py's version of
    this test -- see that file for the full rationale.
    """
    import threading
    import time
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from backend.services.dire_detector import DIREDetector
    from backend.core.model_cache import get_model_cache

    get_model_cache().clear()

    load_calls = []
    call_count_lock = threading.Lock()

    class _FakeScheduler:
        pass

    class _FakePipe:
        def to(self, device):
            return self

    def _slow_fake_scheduler(*a, **kw):
        return _FakeScheduler()

    def _slow_fake_pipeline(*a, **kw):
        with call_count_lock:
            load_calls.append(1)
        time.sleep(0.2)  # hold the window open long enough for threads to race
        return _FakePipe()

    monkeypatch.setattr(DDIMScheduler, "from_pretrained", staticmethod(_slow_fake_scheduler))
    monkeypatch.setattr(StableDiffusionPipeline, "from_pretrained", staticmethod(_slow_fake_pipeline))

    detectors = [DIREDetector() for _ in range(5)]
    threads = [threading.Thread(target=d._load_model) for d in detectors]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(load_calls) == 1, (
        f"expected exactly 1 real model load across 5 concurrent cold-start "
        f"requests for the same cache key, got {len(load_calls)} -- the "
        f"TOCTOU race (F-7) is not closed"
    )
    for d in detectors:
        assert d._model_loaded is True
