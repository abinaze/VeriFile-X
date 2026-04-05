"""
Tests for generator attribution classifier.
All tests must pass without any trained model files present.
"""
import pytest
import numpy as np
from PIL import Image
from io import BytesIO


_VALID_GENERATORS = {"stylegan", "dalle3", "sd14", "sdxl", "midjourney", "real", "unknown"}


def _make_image(width: int = 128, height: int = 128) -> bytes:
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), "RGB"
    )
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_attribution_returns_valid_generator():
    from backend.services.generator_attribution import attribute_generator
    result = attribute_generator(_make_image(), "test.jpg")
    assert result["predicted_generator"] in _VALID_GENERATORS


def test_attribution_confidence_in_range():
    from backend.services.generator_attribution import attribute_generator
    result = attribute_generator(_make_image(), "test.jpg")
    assert 0.0 <= result["confidence"] <= 1.0


def test_attribution_all_scores_present():
    from backend.services.generator_attribution import attribute_generator
    result = attribute_generator(_make_image(), "test.jpg")
    for g in ["stylegan", "dalle3", "sd14", "sdxl", "midjourney", "real"]:
        assert g in result["all_scores"]
        assert 0.0 <= result["all_scores"][g] <= 1.0


def test_attribution_method_field():
    from backend.services.generator_attribution import attribute_generator
    result = attribute_generator(_make_image(), "test.jpg")
    assert result["method"] in ("xgboost", "rule_based", "failed")


def test_attribution_features_extracted():
    from backend.services.generator_attribution import attribute_generator
    result = attribute_generator(_make_image(), "test.jpg")
    assert "mean_hf" in result["features"]
    assert "checker_ratio" in result["features"]
    assert "noise_std" in result["features"]


def test_attribution_handles_small_image():
    from backend.services.generator_attribution import attribute_generator
    result = attribute_generator(_make_image(32, 32), "tiny.jpg")
    assert result["predicted_generator"] in _VALID_GENERATORS


def test_attribution_handles_corrupt_data():
    from backend.services.generator_attribution import attribute_generator
    result = attribute_generator(b"not_an_image", "corrupt.jpg")
    assert result["predicted_generator"] == "unknown"
    assert result["method"] == "failed"


def test_attribution_api_endpoint(client):
    from io import BytesIO
    img_bytes = _make_image()
    response = client.post(
        "/api/v1/analyze/attribution",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_generator" in data
    assert "confidence" in data
    assert "all_scores" in data
    assert "method" in data


def test_attribution_api_rejects_invalid_type(client):
    response = client.post(
        "/api/v1/analyze/attribution",
        files={"file": ("test.txt", b"text", "text/plain")}
    )
    assert response.status_code == 415


def test_attribution_in_forensic_report(client):
    img_bytes = _make_image()
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "generator_attribution" in data
    assert data["generator_attribution"]["predicted_generator"] in _VALID_GENERATORS
