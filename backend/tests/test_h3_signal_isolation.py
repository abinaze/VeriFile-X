"""
Regression tests for H-3 (audit finding): 16 of 19 statistical-bundle
signal methods (BasicSignals x10, UltraSignals x3, CovarianceSignals x3)
had no internal exception handling. A single failing signal propagated
all the way up and discarded the entire ensemble (StatisticalDetector.detect()
has no try/except of its own; ImageForensics.detect_ai_generation()'s
outermost catch-all then replaces the WHOLE 12-signal report with
"analysis_failed", throwing away 11 other signals that had already
succeeded).

Concrete, deterministic evidence a real crash exists (not just a random
fuzz-test hit): a 1x1 pixel image makes analyze_fft_radial_spectrum()
raise ValueError('list argument must have no negative elements') --
confirmed against the real, unpatched method.
"""
import numpy as np
import pytest
from PIL import Image
from io import BytesIO

from backend.services.statistical_signals import (
    BasicSignals,
    UltraSignals,
    CovarianceSignals,
    ImageContext,
    _safe_compute,
)


def _context_for(pixels: np.ndarray, name: str) -> ImageContext:
    img = Image.fromarray(pixels)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return ImageContext.from_bytes(buf.getvalue(), name)


class TestSafeComputeHelper:
    def test_passes_through_successful_result(self):
        result = _safe_compute("X", "x_method", lambda: {"score": 0.7})
        assert result == {"score": 0.7}

    def test_catches_exception_and_returns_neutral_shape(self):
        def boom():
            raise ValueError("synthetic failure")

        result = _safe_compute("Test Signal", "test_method", boom)
        assert result["signal_name"] == "Test Signal"
        assert result["method"] == "test_method"
        assert result["score"] == 0.0
        assert result["confidence"] == 0.0
        assert "raw_value" in result and "expected_range" in result


class TestRealCrashOnDegenerateImage:
    """The concrete, natural trigger: a 1x1 image. Confirmed to raise on
    the real, unpatched analyze_fft_radial_spectrum() before this fix."""

    def test_1x1_image_does_not_crash_basic_signals_compute_all(self):
        ctx = _context_for(np.full((1, 1, 3), 128, dtype=np.uint8), "tiny.png")
        signals = BasicSignals(ctx).compute_all()

        assert len(signals) == 10
        for s in signals:
            assert "score" in s and "confidence" in s and "signal_name" in s

    def test_1x1_image_fft_signal_degrades_to_neutral_not_a_crash(self):
        ctx = _context_for(np.full((1, 1, 3), 128, dtype=np.uint8), "tiny.png")
        signals = BasicSignals(ctx).compute_all()
        fft_signal = next(s for s in signals if s["signal_name"] == "FFT Radial Spectrum")
        assert fft_signal["confidence"] == 0.0
        assert fft_signal["score"] == 0.0


class TestIsolationNotJustSurvival:
    """The important property isn't just "doesn't crash" -- it's that ONE
    failing signal doesn't take down its SIBLINGS in the same bundle."""

    def test_forced_failure_in_one_method_does_not_affect_others(self, monkeypatch):
        ctx = _context_for(
            np.random.default_rng(1).integers(0, 256, (150, 150, 3), dtype=np.uint8),
            "normal.png",
        )
        basic = BasicSignals(ctx)

        def boom():
            raise RuntimeError("forced failure for this test")

        monkeypatch.setattr(basic, "analyze_wavelet_energy", boom)
        signals = basic.compute_all()

        assert len(signals) == 10, "one failing signal must not remove any from the bundle"

        failed = next(s for s in signals if s["signal_name"] == "Wavelet Energy")
        assert failed["confidence"] == 0.0

        others = [s for s in signals if s["signal_name"] != "Wavelet Energy"]
        # The other 9 signals must have actually run and produced normal,
        # non-neutral-fallback confidence values (i.e. they weren't also
        # discarded by the forced failure).
        assert any(s["confidence"] > 0.0 for s in others)


class TestAllThreeBundlesIsolate:
    """Same guarantee, confirmed for UltraSignals and CovarianceSignals too
    -- H-3 affected all three classes, not just BasicSignals."""

    def test_ultra_signals_isolates_a_forced_failure(self, monkeypatch):
        ctx = _context_for(
            np.random.default_rng(2).integers(0, 256, (150, 150, 3), dtype=np.uint8),
            "normal.png",
        )
        ultra = UltraSignals(ctx)
        monkeypatch.setattr(ultra, "analyze_rgb_noise_covariance", lambda: (_ for _ in ()).throw(RuntimeError("x")))
        signals = ultra.compute_all()
        assert len(signals) == 3
        assert signals[0]["confidence"] == 0.0

    def test_covariance_signals_isolates_a_forced_failure(self, monkeypatch):
        ctx = _context_for(
            np.random.default_rng(3).integers(0, 256, (150, 150, 3), dtype=np.uint8),
            "normal.png",
        )
        cov = CovarianceSignals(ctx)
        monkeypatch.setattr(cov, "analyze_eigenvalue_spread", lambda: (_ for _ in ()).throw(RuntimeError("x")))
        signals = cov.compute_all()
        assert len(signals) == 3
        assert signals[0]["confidence"] == 0.0


class TestStatisticalDetectorEndToEnd:
    """The real, end-to-end regression case: StatisticalDetector.detect()
    (no try/except of its own) must not raise on the same 1x1 image that
    crashed the original code, and must still return a genuine 19-signal
    report rather than the outer analysis_failed fallback."""

    def test_detect_survives_1x1_image(self):
        from backend.services.statistical_detector import StatisticalDetector

        img = Image.fromarray(np.full((1, 1, 3), 128, dtype=np.uint8))
        buf = BytesIO()
        img.save(buf, format="PNG")

        detector = StatisticalDetector(buf.getvalue(), "tiny.png")
        report = detector.detect()

        assert report["total_signals"] == 19
        assert 0 <= report["ai_probability"] <= 1
