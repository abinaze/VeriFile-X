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


def test_load_model_is_not_duplicated_under_concurrency(monkeypatch):
    """F-7 regression test: concurrent cold-start requests for the same
    cache key must not each independently trigger a full model load.

    Simulates a slow model load (real CLIP/Stable Diffusion loads take
    seconds to minutes) and fires several concurrent _load_model() calls
    against the same cache key. Before the F-7 fix, every one of them
    would observe a cache miss and every one would call clip.load();
    after the fix, only the first to acquire the per-key lock actually
    loads -- the rest either wait and reuse its result, or (if they
    arrive after it finished) get a cache hit up front.
    """
    import threading
    import time
    import clip
    from backend.services.clip_detector import CLIPDetector
    from backend.core.model_cache import get_model_cache

    get_model_cache().clear()

    load_calls = []
    call_count_lock = threading.Lock()

    class _FakeModel:
        pass

    def _slow_fake_load(name, device=None):
        with call_count_lock:
            load_calls.append(1)
        time.sleep(0.2)  # hold the window open long enough for threads to race
        return _FakeModel(), object()

    monkeypatch.setattr(clip, "load", _slow_fake_load)

    detectors = [CLIPDetector() for _ in range(5)]
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
