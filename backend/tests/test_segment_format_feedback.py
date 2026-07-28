"""
Tests for segment-level AI detection, TIFF/HEIC format support,
and the analyst feedback weight-adaptation system.
"""
import io
import json
import math
import numpy as np
import pytest
from pathlib import Path
from PIL import Image


# ── Image factories ────────────────────────────────────────────────────────────

def _make_jpeg(w=128, h=128):
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_png(w=128, h=128):
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _make_tiff(w=128, h=128):
    rng = np.random.default_rng(13)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="TIFF")
    return buf.getvalue()


def _make_tiny():
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _make_corrupted():
    return b"\xff\xd8\xff\xe0" + b"\x00" * 50


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 27 — Segment-level detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestSegmentDetector:

    def test_returns_required_keys(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_jpeg(), "test.jpg")
        for k in ("grid", "grid_rows", "grid_cols", "tile_size",
                  "stride", "max_score", "mean_score", "hot_tiles", "coverage"):
            assert k in r, f"Missing key: {k}"

    def test_grid_is_2d_list(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_jpeg(), "test.jpg")
        assert isinstance(r["grid"], list)
        assert all(isinstance(row, list) for row in r["grid"])

    def test_grid_dimensions_match(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_jpeg(), "test.jpg")
        assert len(r["grid"]) == r["grid_rows"]
        if r["grid_rows"] > 0:
            assert len(r["grid"][0]) == r["grid_cols"]

    def test_grid_values_in_unit_range(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_jpeg(), "test.jpg")
        for row in r["grid"]:
            for v in row:
                assert 0.0 <= v <= 1.0, f"Grid value out of range: {v}"

    def test_max_score_in_unit_range(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_jpeg(), "test.jpg")
        assert 0.0 <= r["max_score"] <= 1.0

    def test_mean_score_in_unit_range(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_jpeg(), "test.jpg")
        assert 0.0 <= r["mean_score"] <= 1.0

    def test_coverage_in_unit_range(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_jpeg(), "test.jpg")
        assert 0.0 <= r["coverage"] <= 1.0

    def test_tile_size_and_stride_correct(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_jpeg(), "test.jpg")
        assert r["tile_size"] == 64
        assert r["stride"] == 32

    def test_hot_tiles_non_negative(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_jpeg(), "test.jpg")
        assert r["hot_tiles"] >= 0

    def test_tiny_image_returns_fallback(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_tiny(), "tiny.jpg")
        assert r["grid"] == []
        assert r["grid_rows"] == 0

    def test_corrupted_returns_fallback(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_corrupted(), "bad.jpg")
        assert r["grid_rows"] == 0
        assert "error" in r

    def test_png_input_handled(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_png(), "test.png")
        assert isinstance(r["grid"], list)

    def test_no_nan_or_inf(self):
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_jpeg(), "test.jpg")
        for k in ("max_score", "mean_score", "coverage"):
            assert math.isfinite(r[k]), f"Non-finite in '{k}': {r[k]}"

    def test_deterministic(self):
        from backend.services.segment_detector import detect_segments
        b = _make_jpeg(64, 64)
        r1 = detect_segments(b, "det.jpg")
        r2 = detect_segments(b, "det.jpg")
        assert r1["max_score"] == r2["max_score"]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 28 — TIFF and HEIC format support
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatSupport:

    def test_tiff_in_allowed_types(self):
        from backend.core.config import settings
        assert "image/tiff" in settings.ALLOWED_IMAGE_TYPES

    def test_heic_in_allowed_types(self):
        from backend.core.config import settings
        assert "image/heic" in settings.ALLOWED_IMAGE_TYPES

    def test_heif_in_allowed_types(self):
        from backend.core.config import settings
        assert "image/heif" in settings.ALLOWED_IMAGE_TYPES

    def test_tiff_image_opens_correctly(self):
        """PIL must be able to open a TIFF image."""
        tiff_bytes = _make_tiff()
        img = Image.open(io.BytesIO(tiff_bytes))
        assert img.format == "TIFF"

    def test_pillow_heif_registration_in_main(self):
        """main.py must attempt to register pillow_heif."""
        src = (Path(__file__).parent.parent / "main.py").read_text(encoding="utf-8")
        assert "pillow_heif" in src
        assert "register_heif_opener" in src

    def test_pillow_heif_in_requirements(self):
        """pillow-heif must appear in requirements.txt."""
        req = (Path(__file__).parent.parent / "requirements.txt").read_text(encoding="utf-8")
        assert "pillow-heif" in req

    def test_tiff_segment_detection_works(self):
        """segment_detector must handle TIFF input without error."""
        from backend.services.segment_detector import detect_segments
        r = detect_segments(_make_tiff(), "test.tiff")
        assert isinstance(r["grid"], list)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 29 — Analyst feedback weight adaptation
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def tmp_feedback(tmp_path, monkeypatch):
    import backend.services.feedback_manager as fm
    monkeypatch.setattr(fm, "_DATA_DIR",       tmp_path)
    monkeypatch.setattr(fm, "_FEEDBACK_PATH",  tmp_path / "feedback.jsonl")
    monkeypatch.setattr(fm, "_WEIGHTS_PATH",   tmp_path / "signal_weights.json")
    yield fm


def _sample_signals():
    return [
        {"signal_name": "CLIP", "score": 0.8, "confidence": 0.7},
        {"signal_name": "ELA",  "score": 0.75, "confidence": 0.8},
        {"signal_name": "DCT",  "score": 0.6,  "confidence": 0.6},
    ]


class TestFeedbackManager:

    def test_record_returns_required_keys(self, tmp_feedback):
        r = tmp_feedback.record_feedback(
            evidence_id="eid-001",
            true_label="authentic",
            predicted_label="ai_generated",
            signals=_sample_signals(),
        )
        for k in ("feedback_id", "was_wrong", "weights_updated", "total_signals"):
            assert k in r

    def test_wrong_prediction_updates_weights(self, tmp_feedback):
        """Guilty signals must receive a penalty when prediction is wrong."""
        tmp_feedback.record_feedback(
            evidence_id="eid-002",
            true_label="authentic",
            predicted_label="ai_generated",
            signals=_sample_signals(),
        )
        weights = tmp_feedback.load_weights()
        # At least one signal should be penalised
        assert any(v < 1.0 for v in weights.values())

    def test_correct_prediction_no_weight_change(self, tmp_feedback):
        """Correct predictions must not change any weights."""
        tmp_feedback.record_feedback(
            evidence_id="eid-003",
            true_label="ai_generated",
            predicted_label="ai_generated",
            signals=_sample_signals(),
        )
        weights = tmp_feedback.load_weights()
        assert weights == {}  # no updates when prediction was right

    def test_invalid_true_label_raises(self, tmp_feedback):
        with pytest.raises(ValueError, match="true_label"):
            tmp_feedback.record_feedback(
                evidence_id="eid-004",
                true_label="unknown",
                predicted_label="ai_generated",
                signals=[],
            )

    def test_feedback_logged_to_disk(self, tmp_feedback):
        tmp_feedback.record_feedback(
            evidence_id="eid-005",
            true_label="authentic",
            predicted_label="ai_generated",
            signals=_sample_signals(),
        )
        lines = (tmp_feedback._FEEDBACK_PATH).read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["evidence_id"] == "eid-005"

    def test_weight_clipped_to_min(self, tmp_feedback):
        """Weights must never drop below MIN_WEIGHT."""
        # Submit many wrong feedbacks
        for i in range(30):
            tmp_feedback.record_feedback(
                evidence_id=f"eid-{i:03d}",
                true_label="authentic",
                predicted_label="ai_generated",
                signals=[{"signal_name": "CLIP", "score": 0.9, "confidence": 0.9}],
            )
        weights = tmp_feedback.load_weights()
        for v in weights.values():
            assert v >= tmp_feedback._MIN_WEIGHT

    def test_get_feedback_history_empty(self, tmp_feedback):
        assert tmp_feedback.get_feedback_history() == []

    def test_get_feedback_history_returns_records(self, tmp_feedback):
        tmp_feedback.record_feedback(
            evidence_id="eid-006",
            true_label="authentic",
            predicted_label="ai_generated",
            signals=_sample_signals(),
        )
        history = tmp_feedback.get_feedback_history()
        assert len(history) == 1
        assert history[0]["evidence_id"] == "eid-006"

    def test_get_feedback_history_filter_by_id(self, tmp_feedback):
        for eid in ["eid-007", "eid-008", "eid-009"]:
            tmp_feedback.record_feedback(
                evidence_id=eid,
                true_label="authentic",
                predicted_label="ai_generated",
                signals=_sample_signals(),
            )
        filtered = tmp_feedback.get_feedback_history(evidence_id="eid-008")
        assert len(filtered) == 1
        assert filtered[0]["evidence_id"] == "eid-008"

    def test_get_weight_summary_keys(self, tmp_feedback):
        summary = tmp_feedback.get_weight_summary()
        for k in ("signal_weights", "total_overrides",
                  "signals_penalised", "signals_boosted"):
            assert k in summary

    def test_feedback_id_is_unique(self, tmp_feedback):
        import time
        r1 = tmp_feedback.record_feedback(
            "eid-010", "authentic", "ai_generated", _sample_signals()
        )
        time.sleep(0.01)
        r2 = tmp_feedback.record_feedback(
            "eid-011", "authentic", "ai_generated", _sample_signals()
        )
        assert r1["feedback_id"] != r2["feedback_id"]


class TestFeedbackAPIAuth:

    @pytest.fixture()
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api.routes.feedback import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app, raise_server_exceptions=False)

    def test_submit_requires_auth(self, client):
        resp = client.post(
            "/api/v1/feedback/",
            json={
                "evidence_id": "x", "true_label": "authentic",
                "predicted_label": "ai_generated", "signals": [],
            },
        )
        assert resp.status_code == 401

    def test_list_requires_auth(self, client):
        assert client.get("/api/v1/feedback/").status_code == 401

    def test_weights_requires_auth(self, client):
        assert client.get("/api/v1/feedback/weights").status_code == 401
