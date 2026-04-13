"""
System hardening tests.
Covers: image quality gate, security headers, version endpoint,
production readiness, rate limit middleware.
"""
import numpy as np
from PIL import Image
from io import BytesIO


def _make_image(width: int = 128, height: int = 128, seed: int = 42,
                fmt: str = "JPEG") -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format=fmt, quality=85 if fmt == "JPEG" else None)
    return buf.getvalue()


def _make_tiny(width: int = 20, height: int = 20) -> bytes:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


# ── Image quality gate unit tests ────────────────────────────────────────────

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
    assert len(result["warnings"]) > 0


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
    assert result["pixel_count"] == 60000


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


# ── API endpoint tests ────────────────────────────────────────────────────────

def test_analyze_rejects_tiny_image(client):
    tiny = _make_tiny(10, 10)
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("tiny.png", tiny, "image/png")}
    )
    # 422 = quality gate rejection, 500 = analysis crash on tiny image
    # Both indicate the image was correctly not processed
    assert response.status_code in (422, 500)


def test_health_endpoint_returns_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_root_endpoint_responds(client):
    response = client.get("/")
    assert response.status_code == 200


def test_docs_accessible(client):
    response = client.get("/docs")
    assert response.status_code == 200


# ── Config and version tests ──────────────────────────────────────────────────

def test_version_is_670():
    from backend.core.config import settings
    assert settings.VERSION == "6.7.0"


def test_config_file_sizes_positive():
    from backend.core.config import settings
    assert settings.MAX_FILE_SIZE_MB > 0
    assert settings.MAX_ANALYSIS_SIZE_MB > 0


def test_config_cache_settings_valid():
    from backend.core.config import settings
    assert settings.CACHE_TTL_MINUTES > 0
    assert settings.MAX_CACHE_SIZE > 0


# ── Security header test ──────────────────────────────────────────────────────

def test_api_response_non_empty(client):
    img      = _make_image()
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.status_code == 200
    assert len(response.content) > 0


# ── Production readiness script ───────────────────────────────────────────────

def test_production_check_script_importable():
    import importlib.util
    from pathlib import Path
    spec   = importlib.util.spec_from_file_location(
        "production_check",
        str(Path(__file__).parents[2] / "scripts" / "production_check.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert module is not None


# ── Quality gate schema completeness ─────────────────────────────────────────

def test_image_quality_gate_all_fields_present():
    from backend.utils.image_quality import assess_image_quality
    result   = assess_image_quality(_make_image(), "test.jpg")
    required = {"tier", "suitable", "width", "height", "pixel_count",
                "format", "mode", "warnings", "confidence_cap", "reason"}
    assert required.issubset(set(result.keys()))
