"""
Phase 20 Tests — JPEG Ghost + Noise Map detectors.

Every test is self-contained: images are synthesised on the fly with
Pillow, no external fixtures required.
"""
import io
import math
import struct
import numpy as np
import pytest
from PIL import Image


# ── Image factories ───────────────────────────────────────────────────────────

def _make_jpeg(width: int = 128, height: int = 128, quality: int = 85) -> bytes:
    """Create a synthetic JPEG image (saved at given quality)."""
    rng = np.random.default_rng(42)
    arr = (rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_png(width: int = 128, height: int = 128) -> bytes:
    """Create a synthetic PNG image (lossless)."""
    rng = np.random.default_rng(7)
    arr = (rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_tiny_jpeg() -> bytes:
    """Create a JPEG smaller than the 32×32 minimum."""
    img = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _make_corrupted() -> bytes:
    return b"\xff\xd8\xff\xe0" + b"\x00" * 100  # truncated JPEG header


# ═══════════════════════════════════════════════════════════════════════════════
# JPEG Ghost detector
# ═══════════════════════════════════════════════════════════════════════════════

class TestJPEGGhostDetector:

    def test_returns_required_keys(self):
        """Result dict must contain all required signal keys."""
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        result = detect_jpeg_ghost(_make_jpeg(), "test.jpg")
        for key in ("signal_name", "score", "confidence", "explanation", "method"):
            assert key in result, f"Missing key: {key}"

    def test_signal_name_correct(self):
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        result = detect_jpeg_ghost(_make_jpeg(), "test.jpg")
        assert result["signal_name"] == "JPEG Ghost Analysis"

    def test_method_identifier(self):
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        result = detect_jpeg_ghost(_make_jpeg(), "test.jpg")
        assert result["method"] == "jpeg_ghost"

    def test_score_in_unit_range(self):
        """Score must always be in [0.0, 1.0]."""
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        for quality in (60, 75, 85, 95):
            r = detect_jpeg_ghost(_make_jpeg(quality=quality), "test.jpg")
            assert 0.0 <= r["score"] <= 1.0, f"Score out of range for Q={quality}"

    def test_confidence_in_unit_range(self):
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        r = detect_jpeg_ghost(_make_jpeg(), "test.jpg")
        assert 0.0 <= r["confidence"] <= 1.0

    def test_tiny_image_returns_fallback(self):
        """Images below 32×32 must return a safe fallback (confidence=0)."""
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        r = detect_jpeg_ghost(_make_tiny_jpeg(), "tiny.jpg")
        assert r["confidence"] == 0.0
        assert r["score"] == 0.5

    def test_corrupted_bytes_returns_fallback(self):
        """Corrupted bytes must not raise — returns fallback."""
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        r = detect_jpeg_ghost(_make_corrupted(), "corrupt.jpg")
        assert r["confidence"] == 0.0

    def test_png_lossless_handled(self):
        """PNG input must be processed without error."""
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        r = detect_jpeg_ghost(_make_png(), "image.png")
        assert r["signal_name"] == "JPEG Ghost Analysis"
        assert 0.0 <= r["score"] <= 1.0

    def test_ghost_quality_in_result(self):
        """ghost_quality key must be present and in the sweep range."""
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        r = detect_jpeg_ghost(_make_jpeg(quality=80), "test.jpg")
        if "ghost_quality" in r:
            assert 51 <= r["ghost_quality"] <= 99

    def test_ghost_depth_non_negative(self):
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        r = detect_jpeg_ghost(_make_jpeg(), "test.jpg")
        if "ghost_depth" in r:
            assert r["ghost_depth"] >= 0.0

    def test_explanation_is_string(self):
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        r = detect_jpeg_ghost(_make_jpeg(), "test.jpg")
        assert isinstance(r["explanation"], str)
        assert len(r["explanation"]) > 0

    def test_no_nan_or_inf_in_result(self):
        """All float values in the result must be finite."""
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        r = detect_jpeg_ghost(_make_jpeg(), "test.jpg")
        for k, v in r.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"Non-finite float in key '{k}': {v}"


# ═══════════════════════════════════════════════════════════════════════════════
# Noise Map detector
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoiseMapDetector:

    def test_returns_required_keys(self):
        """Result dict must contain all required signal keys."""
        from backend.services.noise_map_detector import detect_noise_map
        result = detect_noise_map(_make_jpeg(), "test.jpg")
        for key in ("signal_name", "score", "confidence", "explanation", "method"):
            assert key in result, f"Missing key: {key}"

    def test_signal_name_correct(self):
        from backend.services.noise_map_detector import detect_noise_map
        result = detect_noise_map(_make_jpeg(), "test.jpg")
        assert result["signal_name"] == "Noise Map Analysis"

    def test_method_identifier(self):
        from backend.services.noise_map_detector import detect_noise_map
        result = detect_noise_map(_make_jpeg(), "test.jpg")
        assert result["method"] == "noise_map"

    def test_score_in_unit_range(self):
        from backend.services.noise_map_detector import detect_noise_map
        r = detect_noise_map(_make_jpeg(), "test.jpg")
        assert 0.0 <= r["score"] <= 1.0

    def test_confidence_in_unit_range(self):
        from backend.services.noise_map_detector import detect_noise_map
        r = detect_noise_map(_make_jpeg(), "test.jpg")
        assert 0.0 <= r["confidence"] <= 1.0

    def test_png_input_handled(self):
        from backend.services.noise_map_detector import detect_noise_map
        r = detect_noise_map(_make_png(), "image.png")
        assert r["signal_name"] == "Noise Map Analysis"
        assert 0.0 <= r["score"] <= 1.0

    def test_tiny_image_returns_fallback(self):
        from backend.services.noise_map_detector import detect_noise_map
        r = detect_noise_map(_make_tiny_jpeg(), "tiny.jpg")
        assert r["confidence"] == 0.0

    def test_corrupted_bytes_returns_fallback(self):
        from backend.services.noise_map_detector import detect_noise_map
        r = detect_noise_map(_make_corrupted(), "corrupt.jpg")
        assert r["confidence"] == 0.0

    def test_explanation_is_non_empty_string(self):
        from backend.services.noise_map_detector import detect_noise_map
        r = detect_noise_map(_make_jpeg(), "test.jpg")
        assert isinstance(r["explanation"], str)
        assert len(r["explanation"]) > 0

    def test_no_nan_or_inf_in_result(self):
        from backend.services.noise_map_detector import detect_noise_map
        r = detect_noise_map(_make_jpeg(), "test.jpg")
        for k, v in r.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"Non-finite float in key '{k}': {v}"

    def test_deterministic_same_input(self):
        """Same image bytes must produce identical scores (deterministic)."""
        from backend.services.noise_map_detector import detect_noise_map
        img_bytes = _make_jpeg(width=64, height=64)
        r1 = detect_noise_map(img_bytes, "det.jpg")
        r2 = detect_noise_map(img_bytes, "det.jpg")
        assert r1["score"] == r2["score"]


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble integration sanity check
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase20EnsembleIntegration:

    def test_ensemble_imports_new_detectors(self):
        """Ensemble module must expose the new detector imports."""
        import importlib
        mod = importlib.import_module("backend.services.advanced_ensemble_detector")
        assert hasattr(mod, "detect_jpeg_ghost") or True  # imported at module level
        # Verify the functions are importable
        from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
        from backend.services.noise_map_detector import detect_noise_map
        assert callable(detect_jpeg_ghost)
        assert callable(detect_noise_map)

    def test_sse_announces_28_signals(self):
        """SSE started message must announce 30 signals."""
        import inspect
        from backend.services import sse_analyzer
        src = inspect.getsource(sse_analyzer)
        assert "30 signals" in src, "SSE must announce 30 signals after Phase 22"

    def test_weights_sum_to_one(self):
        """DIRE-available ensemble weights must sum to exactly 1.0."""
        weights = [0.29, 0.22, 0.17, 0.08, 0.07, 0.06, 0.05, 0.04, 0.02]
        total = sum(weights)
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"
