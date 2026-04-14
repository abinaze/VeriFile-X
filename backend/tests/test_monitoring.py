"""
Tests for Phase 16: observability, inconclusive verdict, image type classifier.
"""
import numpy as np
from PIL import Image
from io import BytesIO


def _make_image(width=128, height=128, seed=42, fmt="JPEG"):
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format=fmt, quality=85 if fmt=="JPEG" else None)
    return buf.getvalue()


def _make_blank():
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


# ── Image type classifier ──────────────────────────────────────────────────────

def test_classifier_returns_valid_type():
    from backend.services.image_type_classifier import classify_image_type
    result = classify_image_type(_make_image(), "test.jpg")
    assert result["image_type"] in {"photo", "screenshot_ui", "illustration",
                                     "document", "low_info", "unknown"}

def test_classifier_blank_is_low_info():
    from backend.services.image_type_classifier import classify_image_type
    result = classify_image_type(_make_blank(), "blank.png")
    assert result["image_type"] == "low_info"
    assert result["run_prnu"] is False
    assert result["run_ela"] is False

def test_classifier_jpeg_photo():
    from backend.services.image_type_classifier import classify_image_type
    result = classify_image_type(_make_image(256, 256, fmt="JPEG"), "photo.jpg")
    assert result["image_type"] in {"photo", "unknown"}
    assert 0.0 <= result["confidence"] <= 1.0

def test_classifier_corrupt_safe():
    from backend.services.image_type_classifier import classify_image_type
    result = classify_image_type(b"corrupt", "bad.jpg")
    assert result["image_type"] == "unknown"

def test_classifier_all_fields_present():
    from backend.services.image_type_classifier import classify_image_type
    result = classify_image_type(_make_image(), "test.jpg")
    for key in ("image_type", "confidence", "is_photo", "run_prnu",
                "run_ela", "run_metadata", "explanation"):
        assert key in result


# ── Metrics collector ─────────────────────────────────────────────────────────

def test_metrics_initial_state():
    from backend.services.metrics_collector import get_metrics, reset_metrics
    reset_metrics()
    m = get_metrics()
    assert m["requests"]["total_in_window"] == 0
    assert m["detection"]["n_scored"] == 0

def test_metrics_record_analysis():
    from backend.services.metrics_collector import record_analysis, get_metrics, reset_metrics
    reset_metrics()
    record_analysis(
        ai_probability=0.85,
        classification="likely_ai_generated",
        latency_ms=250.0,
        predicted_generator="stylegan",
        platform_origin="instagram",
        signal_scores=[{"signal_name": "FFT Radial Spectrum", "score": 0.7}],
    )
    m = get_metrics()
    assert m["requests"]["total_in_window"] == 1
    assert m["detection"]["n_scored"] == 1
    assert m["detection"]["mean_ai_probability"] == 0.85
    assert "likely_ai_generated" in m["classification_breakdown"]
    assert m["classification_breakdown"]["likely_ai_generated"] == 1

def test_metrics_error_recording():
    from backend.services.metrics_collector import record_analysis, get_metrics, reset_metrics
    reset_metrics()
    record_analysis(ai_probability=0.0, classification="error",
                    latency_ms=0.0, error=True)
    m = get_metrics()
    assert m["requests"]["errors_in_window"] == 1
    assert m["requests"]["error_rate"] == 1.0

def test_metrics_reset():
    from backend.services.metrics_collector import record_analysis, reset_metrics, get_metrics
    record_analysis(0.5, "unknown", 100.0)
    reset_metrics()
    m = get_metrics()
    assert m["requests"]["total_in_window"] == 0

def test_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert "requests" in data
    assert "detection" in data
    assert "performance" in data

def test_metrics_reset_endpoint(client):
    response = client.post("/api/v1/metrics/reset")
    assert response.status_code == 200
    assert "reset" in response.json()["message"].lower()


# ── Inconclusive verdict ──────────────────────────────────────────────────────

def test_inconclusive_appears_in_valid_labels():
    """Classification must include inconclusive as a valid label."""
    valid = {"likely_ai_generated", "possibly_ai_generated",
             "possibly_authentic", "likely_authentic", "inconclusive"}
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    det = AdvancedEnsembleDetector(_make_image(), "test.jpg")
    r   = det.detect()
    det.cleanup()
    assert r["classification"] in valid


# ── Image type in forensic report ────────────────────────────────────────────

def test_image_type_in_forensic_report():
    from backend.services.image_forensics import ImageForensics
    rng = np.random.default_rng(seed=5555)
    arr = rng.integers(30, 220, (128, 128, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    report = ImageForensics(buf.getvalue(), "type_test.png").generate_forensic_report()
    assert "image_type" in report
    assert report["image_type"]["image_type"] in {
        "photo", "screenshot_ui", "illustration", "document", "low_info", "unknown"
    }
    assert "image_type" in report["summary"]
