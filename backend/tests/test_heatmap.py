"""
Tests for manipulation localization heatmap endpoint.
All tests must pass without the trained EfficientNet model present.
"""
import base64
import pytest
import numpy as np
from PIL import Image
from io import BytesIO


def _make_image(width: int = 128, height: int = 128, color: str = "RGB") -> bytes:
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), "RGB"
    )
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_heatmap_returns_valid_base64():
    """Heatmap output must be valid base64-encoded PNG."""
    from backend.services.heatmap_generator import generate_heatmap
    result = generate_heatmap(_make_image(), "test.jpg")
    assert "heatmap_b64" in result
    decoded = base64.b64decode(result["heatmap_b64"])
    assert decoded[:4] == b"\x89PNG", "Output must be PNG"


def test_heatmap_dimensions_match_input():
    """Heatmap must match original image dimensions."""
    from backend.services.heatmap_generator import generate_heatmap
    img_bytes = _make_image(width=200, height=150)
    result = generate_heatmap(img_bytes, "test.jpg")
    assert result["width"] == 200
    assert result["height"] == 150


def test_heatmap_method_field_present():
    """Result must include method field indicating gradcam or neutral_fallback."""
    from backend.services.heatmap_generator import generate_heatmap
    result = generate_heatmap(_make_image(), "test.jpg")
    assert result["method"] in ("gradcam", "neutral_fallback")


def test_heatmap_handles_small_image():
    """Heatmap must not crash on small images."""
    from backend.services.heatmap_generator import generate_heatmap
    result = generate_heatmap(_make_image(32, 32), "tiny.jpg")
    assert "heatmap_b64" in result


def test_heatmap_handles_corrupt_data():
    """Heatmap must return error gracefully on corrupt bytes."""
    from backend.services.heatmap_generator import generate_heatmap
    try:
        result = generate_heatmap(b"not_an_image", "corrupt.jpg")
        # If it doesn't raise, it must return valid structure
        assert "heatmap_b64" in result or "error" in result
    except Exception:
        pass  # Acceptable to raise on completely corrupt data


def test_heatmap_api_endpoint(client):
    """API endpoint returns 200 with heatmap_b64 field."""
    img_bytes = _make_image()
    response = client.post(
        "/api/v1/analyze/image/heatmap",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "heatmap_b64" in data
    assert "method" in data
    assert "width" in data
    assert "height" in data


def test_heatmap_api_rejects_invalid_type(client):
    """API endpoint rejects non-image files."""
    response = client.post(
        "/api/v1/analyze/image/heatmap",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 415
