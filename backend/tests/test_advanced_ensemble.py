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


class TestF17ConcurrentSignals:
    """F-17 (partial): the 19-signal statistical bundle + 8 classical
    forensic detectors now run in a thread pool instead of one after
    another. These tests cover what a passing test suite alone won't
    catch for a concurrency change: determinism (no race corrupting a
    result), and that the worker count is capped to the actual core
    count rather than always spinning up 9 threads regardless of what
    hardware is available.

    Profiling this change (see PROFILING_F17.md) found a real trap:
    on a single-core host, 9 threads contending for 1 core measured
    ~1.5x SLOWER than plain sequential execution -- pure context-switch
    overhead with zero real parallelism, not a hypothetical concern.
    The fix caps max_workers at os.cpu_count(), so a single-core
    deployment gracefully falls back to one-thread-at-a-time (matching
    sequential performance) instead of regressing.
    """

    def test_repeated_runs_produce_identical_result(self, sample_image_bytes):
        """The 9 concurrent signals must produce the same combined
        result every time -- if the thread pool were racing on any
        shared state, this would be the first place it would show up
        as a flaky/inconsistent ai_probability across runs."""
        from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector

        results = []
        for _ in range(3):
            detector = AdvancedEnsembleDetector(sample_image_bytes, "test.png")
            report = detector.detect()
            results.append(report["ai_probability"])
            detector.cleanup()

        assert len(set(results)) == 1, (
            f"ai_probability varied across identical repeated runs: {results} "
            "-- possible race condition in the concurrent signal pool"
        )

    def test_max_workers_capped_to_cpu_count(self, sample_image_bytes, monkeypatch):
        """Direct regression test for the single-core slowdown: with
        os.cpu_count() reporting 1, the pool must be constructed with
        max_workers=1, not len(_signal_tasks) (9)."""
        import backend.services.advanced_ensemble_detector as aed

        captured = {}
        real_executor_cls = aed.ThreadPoolExecutor

        class _CapturingExecutor(real_executor_cls):
            def __init__(self, max_workers=None, *a, **kw):
                captured["max_workers"] = max_workers
                super().__init__(max_workers=max_workers, *a, **kw)

        monkeypatch.setattr(aed, "ThreadPoolExecutor", _CapturingExecutor)
        monkeypatch.setattr(aed.os, "cpu_count", lambda: 1)

        detector = aed.AdvancedEnsembleDetector(sample_image_bytes, "test.png")
        detector.detect()
        detector.cleanup()

        assert captured["max_workers"] == 1

    def test_max_workers_never_exceeds_signal_count(self, sample_image_bytes, monkeypatch):
        """On a many-core host, the pool still shouldn't over-allocate
        past the 9 actual tasks -- min(9, cpu_count), not cpu_count."""
        import backend.services.advanced_ensemble_detector as aed

        captured = {}
        real_executor_cls = aed.ThreadPoolExecutor

        class _CapturingExecutor(real_executor_cls):
            def __init__(self, max_workers=None, *a, **kw):
                captured["max_workers"] = max_workers
                super().__init__(max_workers=max_workers, *a, **kw)

        monkeypatch.setattr(aed, "ThreadPoolExecutor", _CapturingExecutor)
        monkeypatch.setattr(aed.os, "cpu_count", lambda: 64)

        detector = aed.AdvancedEnsembleDetector(sample_image_bytes, "test.png")
        detector.detect()
        detector.cleanup()

        assert captured["max_workers"] == 9

    def test_cpu_count_none_treated_as_one_worker(self, sample_image_bytes, monkeypatch):
        """os.cpu_count() can return None on some restricted containers
        -- must not crash, and must be treated the same as a genuinely
        single-core host (max_workers=1), not passed through as None
        (which ThreadPoolExecutor would otherwise interpret as
        "use a default based on os.cpu_count()", silently reintroducing
        the oversubscription this fix exists to prevent)."""
        import backend.services.advanced_ensemble_detector as aed

        monkeypatch.setattr(aed.os, "cpu_count", lambda: None)

        detector = aed.AdvancedEnsembleDetector(sample_image_bytes, "test.png")
        report = detector.detect()  # must not raise
        detector.cleanup()

        assert report["total_signals"] == 30
