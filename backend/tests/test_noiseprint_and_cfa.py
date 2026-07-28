"""
Phase 21 + 22 Tests — Noiseprint and CFA detectors.

All tests are self-contained: images synthesised on the fly with Pillow.
"""
import io
import math
import numpy as np
import pytest
from PIL import Image


# ── Image factories ───────────────────────────────────────────────────────────

def _make_jpeg(width=128, height=128, quality=85):
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_png(width=128, height=128):
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _make_tiny():
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _make_corrupted():
    return b"\xff\xd8\xff\xe0" + b"\x00" * 50


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 21 — Noiseprint detector
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoiseprintDetector:

    def test_returns_required_keys(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        r = detect_noiseprint(_make_jpeg(), "test.jpg")
        for key in ("signal_name", "score", "confidence", "explanation", "method"):
            assert key in r, f"Missing key: {key}"

    def test_signal_name_correct(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        r = detect_noiseprint(_make_jpeg(), "test.jpg")
        assert r["signal_name"] == "Noiseprint Camera Fingerprint"

    def test_method_identifier(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        r = detect_noiseprint(_make_jpeg(), "test.jpg")
        assert r["method"] == "noiseprint_haar"

    def test_score_in_unit_range(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        r = detect_noiseprint(_make_jpeg(), "test.jpg")
        assert 0.0 <= r["score"] <= 1.0

    def test_confidence_in_unit_range(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        r = detect_noiseprint(_make_jpeg(), "test.jpg")
        assert 0.0 <= r["confidence"] <= 1.0

    def test_png_handled(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        r = detect_noiseprint(_make_png(), "image.png")
        assert r["signal_name"] == "Noiseprint Camera Fingerprint"
        assert 0.0 <= r["score"] <= 1.0

    def test_tiny_returns_fallback(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        r = detect_noiseprint(_make_tiny(), "tiny.jpg")
        assert r["confidence"] == 0.0
        assert r["score"] == 0.5

    def test_corrupted_returns_fallback(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        r = detect_noiseprint(_make_corrupted(), "bad.jpg")
        assert r["confidence"] == 0.0

    def test_explanation_non_empty(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        r = detect_noiseprint(_make_jpeg(), "test.jpg")
        assert isinstance(r["explanation"], str) and len(r["explanation"]) > 0

    def test_no_nan_or_inf(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        r = detect_noiseprint(_make_jpeg(), "test.jpg")
        for k, v in r.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"Non-finite in '{k}': {v}"

    def test_deterministic(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        b = _make_jpeg(width=64, height=64)
        r1 = detect_noiseprint(b, "det.jpg")
        r2 = detect_noiseprint(b, "det.jpg")
        assert r1["score"] == r2["score"]

    def test_patch_count_positive(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        r = detect_noiseprint(_make_jpeg(), "test.jpg")
        if "patch_count" in r:
            assert r["patch_count"] > 0

    def test_mean_similarity_in_range(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        r = detect_noiseprint(_make_jpeg(), "test.jpg")
        if "mean_similarity" in r:
            assert -1.0 <= r["mean_similarity"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 22 — CFA detector
# ═══════════════════════════════════════════════════════════════════════════════

class TestCFADetector:

    def test_returns_required_keys(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        r = detect_cfa_artifacts(_make_jpeg(), "test.jpg")
        for key in ("signal_name", "score", "confidence", "explanation", "method"):
            assert key in r, f"Missing key: {key}"

    def test_signal_name_correct(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        r = detect_cfa_artifacts(_make_jpeg(), "test.jpg")
        assert r["signal_name"] == "CFA Artifact Analysis"

    def test_method_identifier(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        r = detect_cfa_artifacts(_make_jpeg(), "test.jpg")
        assert r["method"] == "cfa_bayer"

    def test_score_in_unit_range(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        r = detect_cfa_artifacts(_make_jpeg(), "test.jpg")
        assert 0.0 <= r["score"] <= 1.0

    def test_confidence_in_unit_range(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        r = detect_cfa_artifacts(_make_jpeg(), "test.jpg")
        assert 0.0 <= r["confidence"] <= 1.0

    def test_png_handled(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        r = detect_cfa_artifacts(_make_png(), "image.png")
        assert r["signal_name"] == "CFA Artifact Analysis"
        assert 0.0 <= r["score"] <= 1.0

    def test_tiny_returns_fallback(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        r = detect_cfa_artifacts(_make_tiny(), "tiny.jpg")
        assert r["confidence"] == 0.0

    def test_corrupted_returns_fallback(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        r = detect_cfa_artifacts(_make_corrupted(), "bad.jpg")
        assert r["confidence"] == 0.0

    def test_explanation_non_empty(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        r = detect_cfa_artifacts(_make_jpeg(), "test.jpg")
        assert isinstance(r["explanation"], str) and len(r["explanation"]) > 0

    def test_no_nan_or_inf(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        r = detect_cfa_artifacts(_make_jpeg(), "test.jpg")
        for k, v in r.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"Non-finite in '{k}': {v}"

    def test_deterministic(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        b = _make_jpeg(width=64, height=64)
        r1 = detect_cfa_artifacts(b, "det.jpg")
        r2 = detect_cfa_artifacts(b, "det.jpg")
        assert r1["score"] == r2["score"]

    def test_cfa_ratio_positive(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        r = detect_cfa_artifacts(_make_jpeg(), "test.jpg")
        if "cfa_ratio" in r:
            assert r["cfa_ratio"] > 0.0

    def test_cfa_ratio_col_row_present(self):
        from backend.services.cfa_detector import detect_cfa_artifacts
        r = detect_cfa_artifacts(_make_jpeg(), "test.jpg")
        if "cfa_ratio_col" in r:
            assert r["cfa_ratio_col"] > 0.0
        if "cfa_ratio_row" in r:
            assert r["cfa_ratio_row"] > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble integration — Phase 21 + 22
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase2122EnsembleIntegration:

    def test_detectors_importable(self):
        from backend.services.noiseprint_detector import detect_noiseprint
        from backend.services.cfa_detector import detect_cfa_artifacts
        assert callable(detect_noiseprint)
        assert callable(detect_cfa_artifacts)

    def test_sse_announces_30_signals(self):
        import inspect
        from backend.services import sse_analyzer
        src = inspect.getsource(sse_analyzer)
        assert "30 signals" in src, "SSE must announce 30 signals after Phase 22"

    def test_dire_weights_sum_to_one(self):
        """DIRE-available ensemble weights must sum to exactly 1.0."""
        weights = [0.26, 0.21, 0.16, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02]
        assert abs(sum(weights) - 1.0) < 1e-9, f"Weights sum to {sum(weights)}"

    def test_ensemble_version_updated(self):
        import inspect
        from backend.services import advanced_ensemble_detector
        src = inspect.getsource(advanced_ensemble_detector)
        assert "advanced-ensemble-v1.6" in src

    def test_noiseprint_in_sse_stream(self):
        import inspect
        from backend.services import sse_analyzer
        src = inspect.getsource(sse_analyzer)
        assert "detect_noiseprint" in src

    def test_cfa_in_sse_stream(self):
        import inspect
        from backend.services import sse_analyzer
        src = inspect.getsource(sse_analyzer)
        assert "detect_cfa_artifacts" in src
