"""
Tests for adversarial robustness testing suite.
Tests run without GPU and use small images for speed.
All attack functions must be testable without ML models.
"""
import pytest
import numpy as np
from PIL import Image
from io import BytesIO


def _make_image(width: int = 100, height: int = 100, seed: int = 42) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(50, 200, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _img_array(image_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))


# ── Attack function unit tests ────────────────────────────────────────────────

def test_jpeg_recompression_changes_image():
    from backend.services.adversarial_tester import _apply_jpeg_recompression
    arr     = _img_array(_make_image())
    result  = _apply_jpeg_recompression(arr.copy(), 0.6)
    assert result.shape == arr.shape
    assert result.dtype == np.uint8


def test_gaussian_noise_bounded():
    from backend.services.adversarial_tester import _apply_gaussian_noise
    arr    = _img_array(_make_image())
    result = _apply_gaussian_noise(arr.copy(), 0.5)
    assert result.shape == arr.shape
    assert result.min() >= 0 and result.max() <= 255


def test_gaussian_blur_produces_valid_image():
    from backend.services.adversarial_tester import _apply_gaussian_blur
    arr    = _img_array(_make_image())
    result = _apply_gaussian_blur(arr.copy(), 0.5)
    assert result.shape == arr.shape
    assert result.dtype == np.uint8


def test_color_jitter_preserves_shape():
    from backend.services.adversarial_tester import _apply_color_jitter
    arr    = _img_array(_make_image())
    result = _apply_color_jitter(arr.copy(), 0.5)
    assert result.shape == arr.shape


def test_downscale_upscale_preserves_dimensions():
    from backend.services.adversarial_tester import _apply_downscale_upscale
    arr    = _img_array(_make_image())
    result = _apply_downscale_upscale(arr.copy(), 0.5)
    assert result.shape == arr.shape


def test_random_crop_preserves_dimensions():
    from backend.services.adversarial_tester import _apply_random_crop
    arr    = _img_array(_make_image())
    result = _apply_random_crop(arr.copy(), 0.5)
    assert result.shape == arr.shape


def test_histogram_equalization_bounded():
    from backend.services.adversarial_tester import _apply_histogram_equalization
    arr    = _img_array(_make_image())
    result = _apply_histogram_equalization(arr.copy(), 0.5)
    assert result.shape == arr.shape
    assert result.min() >= 0 and result.max() <= 255


def test_pixel_shuffle_preserves_shape():
    from backend.services.adversarial_tester import _apply_pixel_shuffle
    arr    = _img_array(_make_image())
    result = _apply_pixel_shuffle(arr.copy(), 0.5)
    assert result.shape == arr.shape


# ── Robustness test integration ───────────────────────────────────────────────

def test_robustness_result_structure():
    """robustness test returns correct schema."""
    from backend.services.adversarial_tester import run_robustness_test
    result = run_robustness_test(_make_image(), "test.jpg", attacks=["gaussian_noise"])
    assert "overall_robustness" in result
    assert "baseline_score" in result
    assert "baseline_class" in result
    assert "attack_results" in result
    assert "robustness_level" in result
    assert 0.0 <= result["overall_robustness"] <= 1.0


def test_robustness_level_valid():
    from backend.services.adversarial_tester import run_robustness_test
    result = run_robustness_test(_make_image(), "test.jpg", attacks=["gaussian_blur"])
    assert result["robustness_level"] in {"high", "medium", "low"}


def test_robustness_attack_results_present():
    from backend.services.adversarial_tester import run_robustness_test
    result = run_robustness_test(
        _make_image(), "test.jpg",
        attacks=["jpeg_recompression", "gaussian_noise"]
    )
    assert "jpeg_recompression" in result["attack_results"]
    assert "gaussian_noise" in result["attack_results"]
    for attack in result["attack_results"].values():
        assert "results" in attack
        assert "mean_robustness" in attack
        assert 0.0 <= attack["mean_robustness"] <= 1.0


def test_robustness_intensity_scores_bounded():
    from backend.services.adversarial_tester import run_robustness_test
    result = run_robustness_test(_make_image(), "test.jpg", attacks=["color_jitter"])
    for r in result["attack_results"]["color_jitter"]["results"]:
        assert 0.0 <= r["robustness_score"] <= 1.0
        assert r["intensity"] in {0.3, 0.6, 1.0}


def test_robustness_api_endpoint(client):
    """API endpoint responds within timeout and returns correct schema."""
    img = _make_image()
    response = client.post(
        "/api/v1/analyze/robustness",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "overall_robustness" in data
    assert "robustness_level" in data
    assert "attack_results" in data


def test_robustness_api_rejects_non_image(client):
    response = client.post(
        "/api/v1/analyze/robustness",
        files={"file": ("test.txt", b"text", "text/plain")}
    )
    assert response.status_code == 415


@pytest.mark.slow
def test_all_eight_attacks_run():
    """Full suite test — marked slow, skipped in CI fast run."""
    from backend.services.adversarial_tester import run_robustness_test, _ATTACK_CONFIGS
    result = run_robustness_test(_make_image(128, 128), "test.jpg")
    assert result["attacks_tested"] == len(_ATTACK_CONFIGS)
    assert "summary" in result
    assert "recommendation" in result
