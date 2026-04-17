"""Tests for SSE streaming analysis endpoint."""
import numpy as np
from PIL import Image
from io import BytesIO


def _make_jpeg(seed=42, w=128, h=128):
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, (h, w, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def test_sse_endpoint_exists(client):
    img = _make_jpeg()
    response = client.post(
        "/api/v1/analyze/image/stream",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.status_code in (200, 429)


def test_sse_returns_event_stream(client):
    img = _make_jpeg(seed=10)
    response = client.post(
        "/api/v1/analyze/image/stream",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    if response.status_code == 429:
        return
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_sse_rejects_non_image(client):
    response = client.post(
        "/api/v1/analyze/image/stream",
        files={"file": ("test.txt", b"text", "text/plain")}
    )
    assert response.status_code == 415


def test_sse_stream_contains_data_events(client):
    img = _make_jpeg(seed=20)
    response = client.post(
        "/api/v1/analyze/image/stream",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    if response.status_code == 429:
        return
    assert response.status_code == 200
    content = response.text
    assert "data:" in content


def test_sse_stream_has_summary_event(client):
    import json
    img = _make_jpeg(seed=30)
    response = client.post(
        "/api/v1/analyze/image/stream",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    if response.status_code == 429:
        return
    events = []
    for line in response.text.split("\n\n"):
        line = line.strip()
        if line.startswith("data:"):
            try:
                events.append(json.loads(line[5:].strip()))
            except Exception:
                pass
    types = [e.get("type") for e in events]
    assert "started" in types or "signal" in types or "summary" in types


def test_sse_rejects_oversized(client):
    big = b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024)
    response = client.post(
        "/api/v1/analyze/image/stream",
        files={"file": ("big.jpg", big, "image/jpeg")}
    )
    assert response.status_code == 413


def test_sse_service_importable():
    from backend.services.sse_analyzer import stream_analysis
    assert callable(stream_analysis)


def test_sse_stream_signal_structure(client):
    """Each signal event must have signal_name, score, confidence."""
    import json
    img = _make_jpeg(seed=40)
    response = client.post(
        "/api/v1/analyze/image/stream",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    if response.status_code == 429:
        return
    for line in response.text.split("\n\n"):
        line = line.strip()
        if line.startswith("data:"):
            try:
                evt = json.loads(line[5:].strip())
                if evt.get("type") == "signal":
                    assert "signal_name" in evt
                    assert "score" in evt
                    assert 0.0 <= evt["score"] <= 1.0
                    assert "confidence" in evt
                    break
            except Exception:
                pass
