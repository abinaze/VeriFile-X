"""Tests for social media platform signature detection."""
import pytest
import numpy as np
from PIL import Image
from io import BytesIO

_VALID_PLATFORMS = {
    "whatsapp", "instagram", "discord", "telegram",
    "twitter_x", "facebook", "original", "unknown"
}


def _make_jpeg(quality: int = 85, width: int = 128, height: int = 128) -> bytes:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_png(width: int = 128, height: int = 128) -> bytes:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_platform_returns_valid_label():
    from backend.services.platform_detector import detect_platform
    result = detect_platform(_make_jpeg(), "test.jpg")
    assert result["predicted_platform"] in _VALID_PLATFORMS


def test_platform_confidence_in_range():
    from backend.services.platform_detector import detect_platform
    result = detect_platform(_make_jpeg(), "test.jpg")
    assert 0.0 <= result["confidence"] <= 1.0


def test_platform_all_scores_present():
    from backend.services.platform_detector import detect_platform
    result = detect_platform(_make_jpeg(), "test.jpg")
    for p in ["whatsapp", "instagram", "discord", "telegram",
              "twitter_x", "facebook", "original"]:
        assert p in result["all_scores"]


def test_png_detected_as_original():
    """PNG images should be classified as original (lossless)."""
    from backend.services.platform_detector import detect_platform
    result = detect_platform(_make_png(), "test.png")
    assert result["predicted_platform"] == "original"
    assert result["confidence"] >= 0.5


def test_features_extracted():
    from backend.services.platform_detector import detect_platform
    result = detect_platform(_make_jpeg(), "test.jpg")
    assert "estimated_quality" in result["features"]
    assert "has_exif" in result["features"]
    assert "hf_ratio" in result["features"]


def test_corrupt_input_returns_unknown():
    from backend.services.platform_detector import detect_platform
    result = detect_platform(b"not_an_image", "corrupt.bin")
    assert result["predicted_platform"] == "unknown"
    assert result["confidence"] == 0.0


def test_platform_api_endpoint(client):
    img = _make_jpeg()
    response = client.post(
        "/api/v1/analyze/platform",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_platform" in data
    assert "confidence" in data
    assert "all_scores" in data


def test_platform_api_rejects_invalid_type(client):
    response = client.post(
        "/api/v1/analyze/platform",
        files={"file": ("test.txt", b"text", "text/plain")}
    )
    assert response.status_code == 415


def test_platform_in_forensic_report(client):
    # Use unique image to avoid cache hit from previous tests
    import numpy as np
    from PIL import Image
    from io import BytesIO
    rng = np.random.default_rng(seed=9999)
    arr = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    fresh_bytes = buf.getvalue()
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("platform_test_unique.png", fresh_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "platform_forensics" in data
    assert data["platform_forensics"]["predicted_platform"] in _VALID_PLATFORMS
    assert "platform_origin" in data["summary"]
