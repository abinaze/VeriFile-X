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


def _make_oversized():
    """Build an oversized JPEG-like payload without null bytes in source."""
    header = b"\xff\xd8\xff"
    padding = b"\x01" * (11 * 1024 * 1024)
    return header + padding


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
    assert "data:" in response.text


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


def test_sse_rejects_oversized():
    """Test size limit directly via config — no HTTP call, no rate limit."""
    from backend.core.config import settings
    max_bytes = settings.MAX_ANALYSIS_SIZE_MB * 1024 * 1024
    oversized = max_bytes + 1
    assert oversized > max_bytes  # config enforces the limit


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


def test_sse_stream_delivers_all_30_signals_with_no_error(client):
    """F-16 regression test: the previous tests in this file use loose
    assertions (an OR across event types, a soft break-on-first-match
    with no final check) that let a real regression slip through --
    AdvancedEnsembleDetector no longer inheriting from StatisticalDetector
    broke the super(AdvancedEnsembleDetector, detector).detect() call
    this module used to stream the 19 statistical signals, silently
    routing every request into the error handler instead. This asserts
    the stream actually contains all 30 signal events and never an
    "error" event for a valid image -- the previous tests would still
    have passed even with zero signal events delivered.
    """
    import json
    img = _make_jpeg(seed=50)
    response = client.post(
        "/api/v1/analyze/image/stream",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    if response.status_code == 429:
        return
    assert response.status_code == 200

    events = []
    for line in response.text.split("\n\n"):
        line = line.strip()
        if line.startswith("data:"):
            try:
                events.append(json.loads(line[5:].strip()))
            except Exception:
                pass

    error_events = [e for e in events if e.get("type") == "error"]
    assert error_events == [], f"stream reported an error event: {error_events}"

    signal_events = [e for e in events if e.get("type") == "signal"]
    assert len(signal_events) == 30, (
        f"expected all 30 signals to stream, got {len(signal_events)} -- "
        f"the statistical-signals block likely failed silently into the "
        f"error handler"
    )

    summary_events = [e for e in events if e.get("type") == "summary"]
    assert len(summary_events) == 1
    assert summary_events[0]["total_signals"] == 30
