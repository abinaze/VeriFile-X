"""
System hardening tests — quality gate, version, security, production readiness.
"""
import numpy as np
from PIL import Image
from io import BytesIO


def _make_image(width=128, height=128, seed=42, fmt="JPEG"):
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format=fmt, quality=85 if fmt == "JPEG" else None)
    return buf.getvalue()


def _make_tiny(width=20, height=20):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_quality_gate_good_image():
    from backend.utils.image_quality import assess_image_quality
    result = assess_image_quality(_make_image(256, 256), "test.jpg")
    assert result["suitable"] is True
    assert result["tier"] == "good"
    assert result["confidence_cap"] == 1.0


def test_quality_gate_rejects_tiny_image():
    from backend.utils.image_quality import assess_image_quality
    result = assess_image_quality(_make_tiny(20, 20), "tiny.png")
    assert result["suitable"] is False
    assert result["tier"] == "unsuitable"
    assert result["confidence_cap"] == 0.0
    assert result["reason"] is not None


def test_quality_gate_warns_small_image():
    from backend.utils.image_quality import assess_image_quality
    result = assess_image_quality(_make_image(100, 100), "small.jpg")
    assert result["suitable"] is True
    assert result["tier"] in ("low", "degraded")
    assert result["confidence_cap"] < 1.0


def test_quality_gate_corrupt_bytes():
    from backend.utils.image_quality import assess_image_quality
    result = assess_image_quality(b"not_an_image", "corrupt.bin")
    assert result["suitable"] is False
    assert result["tier"] == "unsuitable"


def test_quality_gate_returns_dimensions():
    from backend.utils.image_quality import assess_image_quality
    result = assess_image_quality(_make_image(300, 200), "test.jpg")
    assert result["width"] == 300
    assert result["height"] == 200


def test_quality_gate_png_accepted():
    from backend.utils.image_quality import assess_image_quality
    result = assess_image_quality(_make_image(256, 256, fmt="PNG"), "test.png")
    assert result["suitable"] is True
    assert result["format"] == "PNG"


def test_quality_gate_grayscale_warns():
    from backend.utils.image_quality import assess_image_quality
    buf = BytesIO()
    Image.fromarray(np.zeros((128, 128), dtype=np.uint8), "L").save(buf, format="PNG")
    result = assess_image_quality(buf.getvalue(), "gray.png")
    assert result["suitable"] is True
    assert any("grayscale" in w.lower() or "Grayscale" in w for w in result["warnings"])


def test_analyze_rejects_tiny_image(client):
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("tiny.png", _make_tiny(10, 10), "image/png")}
    )
    assert response.status_code == 422


def _fake_report_with_nan(evidence_id="evid-f8-test"):
    """Minimal report shape matching everything analyze_image() reads,
    with a NaN planted in a signal score -- exactly the kind of value a
    division-by-zero guard elsewhere in the ensemble can legitimately
    produce."""
    return {
        "evidence_id": evidence_id,
        "summary": {
            "ai_probability": float("nan"),
            "ai_classification": "uncertain",
            "total_detection_signals": 1,
            "suspicious_detection_signals": 0,
        },
        "ai_detection": {
            "methods_used": ["statistical analysis"],
            "all_signals": [
                {"signal_name": "test_signal", "score": float("nan"),
                 "confidence": 0.9, "explanation": "x", "raw_value": float("inf"),
                 "expected_range": "0-1"},
            ],
        },
    }


def test_f8_sanitize_runs_before_cache_webhook_and_audit_log(client, monkeypatch):
    """F-8 regression test: the cache entry, the outbound webhook payload,
    and the audit log must all see a sanitized report -- not just the
    direct HTTP response.

    Before the fix, forensics_cache.set() / fire_webhooks() / log_analysis()
    ran on the raw report (still containing NaN/Infinity), and only the
    return statement's late _sanitize() call protected the HTTP response
    itself. This monkeypatches report generation to return a report with
    a planted NaN/Infinity, then inspects what the cache and the webhook
    call actually received.
    """
    import backend.services.image_forensics as forensics_mod
    import backend.api.routes.analyze as analyze_mod
    from backend.core.cache import forensics_cache

    # This test calls /api/v1/analyze/image, which is rate-limited to
    # 10/minute per client key. slowapi's default in-memory storage is
    # shared for the whole pytest session (no reset between tests), so
    # without this reset this test's call would consume budget from --
    # or be starved by -- unrelated tests elsewhere in the suite that
    # also hit this endpoint (confirmed while writing this test: running
    # the full suite made test_performance.py::test_cache_speedup_is_significant
    # flake with a 429, purely from accumulated cross-test call volume).
    analyze_mod.limiter.limiter.storage.reset()

    monkeypatch.setattr(
        forensics_mod.ImageForensics, "generate_forensic_report",
        lambda self: _fake_report_with_nan(),
    )

    webhook_payloads = []
    monkeypatch.setattr(
        "backend.services.webhook_manager.fire_webhooks",
        lambda evidence_id, result, event: webhook_payloads.append(result),
    )

    # Capture exactly what gets written to the cache, rather than trying to
    # independently recompute the file hash analyze_image() uses internally
    # (it hashes the image AFTER an EXIF-transpose re-encode step, which is
    # an implementation detail this test shouldn't need to replicate).
    cache_writes = []
    _real_cache_set = forensics_cache.set
    def _spy_cache_set(key, value):
        cache_writes.append(value)
        return _real_cache_set(key, value)
    monkeypatch.setattr(forensics_cache, "set", _spy_cache_set)

    image_bytes = _make_image(256, 256, seed=99)
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("f8_test.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200

    # 1) The HTTP response itself must be clean (this part already worked
    #    before the fix, since JSON encoding NaN/Infinity would otherwise
    #    either fail or emit invalid tokens).
    body = response.json()
    assert body["summary"]["ai_probability"] == 0.0
    assert body["ai_detection"]["all_signals"][0]["score"] == 0.0
    assert body["ai_detection"]["all_signals"][0]["raw_value"] == 0.0

    # 2) The webhook payload must ALSO be sanitized -- this is what F-8
    #    actually fixes (it was NOT true before the fix).
    assert len(webhook_payloads) == 1
    assert webhook_payloads[0]["summary"]["ai_probability"] == 0.0
    assert webhook_payloads[0]["ai_detection"]["all_signals"][0]["score"] == 0.0

    # 3) What actually gets written to the cache must ALSO be sanitized --
    #    the cache-hit branch no longer runs its own separate sanitize copy
    #    at all (F-27), so it can only return clean data if what was
    #    stored was already clean.
    assert len(cache_writes) == 1
    assert cache_writes[0]["summary"]["ai_probability"] == 0.0
    assert cache_writes[0]["ai_detection"]["all_signals"][0]["score"] == 0.0

    # Leave the shared limiter clean for whatever test runs next.
    analyze_mod.limiter.limiter.storage.reset()


def test_health_endpoint_returns_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")  # degraded in CI without model files
    # Verify the new degraded_detectors field is always present (may be empty list)
    assert "degraded_detectors" in data, "Health response must include degraded_detectors list"
    assert isinstance(data["degraded_detectors"], list)


def test_root_endpoint_responds(client):
    assert client.get("/").status_code == 200


def test_docs_accessible(client):
    assert client.get("/docs").status_code == 200


def test_version_is_850():
    from backend.core.config import settings
    assert settings.VERSION == "8.5.0"


def test_config_file_sizes_positive():
    from backend.core.config import settings
    assert settings.MAX_FILE_SIZE_MB > 0
    assert settings.MAX_ANALYSIS_SIZE_MB > 0


def test_config_cache_settings_valid():
    from backend.core.config import settings
    assert settings.CACHE_TTL_MINUTES > 0
    assert settings.MAX_CACHE_SIZE > 0


def test_api_response_non_empty(client):
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("test.jpg", _make_image(), "image/jpeg")}
    )
    assert response.status_code == 200
    assert len(response.content) > 0


def test_production_check_importable():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "production_check",
        str(Path(__file__).parents[2] / "scripts" / "production_check.py")
    )
    assert importlib.util.module_from_spec(spec) is not None


def test_quality_gate_all_fields_present():
    from backend.utils.image_quality import assess_image_quality
    result = assess_image_quality(_make_image(), "test.jpg")
    required = {"tier", "suitable", "width", "height", "pixel_count",
                "format", "mode", "warnings", "confidence_cap", "reason"}
    assert required.issubset(set(result.keys()))
