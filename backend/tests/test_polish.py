"""Phase 17 — Final polish tests."""
import numpy as np
from PIL import Image
from io import BytesIO


def _make_image(width=128, height=128, seed=99):
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def test_version_is_700():
    from backend.core.config import settings
    assert settings.VERSION == "7.0.0"


def test_metrics_endpoint_schema(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    for key in ("uptime_seconds", "uptime_human", "timestamp",
                "requests", "detection", "performance",
                "classification_breakdown"):
        assert key in data, f"Missing key: {key}"


def test_metrics_detection_keys(client):
    response = client.get("/api/v1/metrics")
    det = response.json()["detection"]
    for key in ("mean_ai_probability", "median_ai_probability",
                "ai_positive_rate", "n_scored"):
        assert key in det


def test_metrics_requests_keys(client):
    response = client.get("/api/v1/metrics")
    req = response.json()["requests"]
    for key in ("total_in_window", "errors_in_window",
                "requests_last_60s", "error_rate"):
        assert key in req


def test_metrics_performance_keys(client):
    response = client.get("/api/v1/metrics")
    perf = response.json()["performance"]
    for key in ("mean_latency_ms", "p95_latency_ms", "n_timed"):
        assert key in perf


def test_metrics_reset_clears_data(client):
    r = client.post("/api/v1/metrics/reset")
    assert r.status_code == 200
    m = client.get("/api/v1/metrics").json()
    assert m["detection"]["n_scored"] == 0


def test_image_type_in_analyze_report(client):
    img = _make_image(seed=777)
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    if response.status_code == 429:
        return
    assert response.status_code == 200
    data = response.json()
    assert "image_type" in data
    assert "image_type" in data["summary"]


def test_inconclusive_is_valid_classification(client):
    valid = {"likely_ai_generated", "possibly_ai_generated",
             "possibly_authentic", "likely_authentic", "inconclusive"}
    img = _make_image(seed=888)
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    if response.status_code == 429:
        return
    assert response.status_code == 200
    assert response.json()["summary"]["ai_classification"] in valid


def test_analyze_report_no_nan_values(client):
    import math, json
    img = _make_image(seed=666)
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    if response.status_code == 429:
        return
    assert response.status_code == 200

    def check_no_nan(obj, path=""):
        if isinstance(obj, float):
            assert math.isfinite(obj), f"NaN/Inf at {path}: {obj}"
        elif isinstance(obj, dict):
            for k, v in obj.items(): check_no_nan(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj): check_no_nan(v, f"{path}[{i}]")
    check_no_nan(json.loads(response.text))


def test_shared_limiter_exists():
    from backend.main import shared_limiter
    assert shared_limiter is not None


def test_metrics_uptime_positive():
    from backend.services.metrics_collector import get_metrics
    m = get_metrics()
    assert m["uptime_seconds"] >= 0
    assert len(m["uptime_human"]) > 0
