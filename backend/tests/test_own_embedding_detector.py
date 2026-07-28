"""
Tests for OwnEmbeddingDetector's model caching, including the F-7
concurrency regression test (no prior test file covered this detector).
"""
import threading
import time

import pytest


def test_load_model_uses_shared_cache(monkeypatch):
    """A second detector instance must reuse the first's cached model
    rather than reloading from disk."""
    from backend.services.own_embedding_detector import OwnEmbeddingDetector, _CACHE_KEY
    from backend.core.model_cache import get_model_cache
    import backend.services.own_detector.model as model_mod

    get_model_cache().clear()

    load_calls = []

    class _FakeModel:
        def eval(self):
            pass

    def _fake_load_model(device):
        load_calls.append(1)
        return _FakeModel()

    monkeypatch.setattr(model_mod, "load_model", _fake_load_model)

    d1 = OwnEmbeddingDetector()
    d1._load_model()
    d2 = OwnEmbeddingDetector()
    d2._load_model()

    assert len(load_calls) == 1, "second detector should reuse the cached model, not reload"
    assert d2._model_loaded is True


def test_load_model_is_not_duplicated_under_concurrency(monkeypatch):
    """F-7 regression test: concurrent cold-start requests must not each
    independently load the model. Same shape as the equivalent tests for
    clip_detector.py and dire_detector.py.
    """
    from backend.services.own_embedding_detector import OwnEmbeddingDetector, _CACHE_KEY
    from backend.core.model_cache import get_model_cache
    import backend.services.own_detector.model as model_mod

    get_model_cache().clear()

    load_calls = []
    call_count_lock = threading.Lock()

    class _FakeModel:
        def eval(self):
            pass

    def _slow_fake_load_model(device):
        with call_count_lock:
            load_calls.append(1)
        time.sleep(0.2)  # hold the window open long enough for threads to race
        return _FakeModel()

    monkeypatch.setattr(model_mod, "load_model", _slow_fake_load_model)

    detectors = [OwnEmbeddingDetector() for _ in range(5)]
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
