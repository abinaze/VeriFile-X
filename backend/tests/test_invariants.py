"""
Invariant and property-based tests for VeriFile-X detection pipeline.
"""
import pytest
import numpy as np
from PIL import Image
from io import BytesIO


def _make_image(width=128, height=128, seed=42) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_noise_image(width=128, height=128) -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _encode(arr: np.ndarray, fmt: str, quality: int = 85) -> bytes:
    buf = BytesIO()
    img = Image.fromarray(arr.astype(np.uint8), "RGB")
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def test_detection_repeatable_10_runs(sample_image_bytes):
    """Same image must produce identical scores across 10 consecutive runs."""
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    scores = []
    for _ in range(10):
        det = AdvancedEnsembleDetector(sample_image_bytes, "repeat_test.png")
        r   = det.detect()
        det.cleanup()
        scores.append(round(r["ai_probability"], 8))
    assert len(set(scores)) == 1, f"Non-deterministic scores: {set(scores)}"


def test_attribution_repeatable(sample_image_bytes):
    """Generator attribution must be stable across repeated calls."""
    from backend.services.generator_attribution import attribute_generator
    results = [
        attribute_generator(sample_image_bytes, "test.jpg")["predicted_generator"]
        for _ in range(5)
    ]
    assert len(set(results)) == 1, f"Attribution not stable: {set(results)}"


def test_noise_image_score_bounded():
    """Pure noise image must produce a score in [0.0, 1.0]."""
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    det   = AdvancedEnsembleDetector(_make_noise_image(), "noise.jpg")
    r     = det.detect()
    det.cleanup()
    assert 0.0 <= r["ai_probability"] <= 1.0


def test_noise_image_signal_count():
    """Every signal must be present even for pure noise input."""
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    det = AdvancedEnsembleDetector(_make_noise_image(), "noise.jpg")
    r   = det.detect()
    det.cleanup()
    assert r["total_signals"] == 26
    for sig in r["all_signals"]:
        assert 0.0 <= sig["score"] <= 1.0
        assert 0.0 <= sig["confidence"] <= 1.0


def test_noise_image_attribution_valid():
    """Attribution must not crash on pure noise."""
    from backend.services.generator_attribution import attribute_generator
    result = attribute_generator(_make_noise_image(), "noise.jpg")
    assert result["predicted_generator"] in {
        "stylegan", "dalle3", "sd14", "sdxl", "midjourney", "real", "unknown"
    }
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.parametrize("bad_input", [
    b"",
    b"\xff\xd8\xff",
    b"PNG\r\n\x1a\n" + b"\x00" * 10,
    bytes(range(256)),
    b"\x00" * 1024,
])
def test_corrupt_input_does_not_crash(bad_input):
    """System must never raise an unhandled exception on malformed input."""
    from backend.services.generator_attribution import attribute_generator
    from backend.services.heatmap_generator import generate_heatmap

    try:
        result = attribute_generator(bad_input, "corrupt.bin")
        assert "predicted_generator" in result
    except (ValueError, IOError, OSError):
        pass

    try:
        result = generate_heatmap(bad_input, "corrupt.bin")
        assert "heatmap_b64" in result
    except (ValueError, IOError, OSError):
        pass


def test_jpeg_png_attribution_consistent():
    """JPEG and PNG of same image must produce same top-2 generators."""
    from backend.services.generator_attribution import attribute_generator
    rng = np.random.default_rng(7)
    arr = rng.integers(30, 220, (128, 128, 3), dtype=np.uint8)
    r_jpeg = attribute_generator(_encode(arr, "JPEG", 95), "test.jpg")
    r_png  = attribute_generator(_encode(arr, "PNG"),      "test.png")
    top2_jpeg = sorted(r_jpeg["all_scores"], key=r_jpeg["all_scores"].get, reverse=True)[:2]
    top2_png  = sorted(r_png["all_scores"],  key=r_png["all_scores"].get,  reverse=True)[:2]
    assert set(top2_jpeg) == set(top2_png), (
        f"Top-2 differ: JPEG={top2_jpeg}, PNG={top2_png}"
    )


def test_signal_scores_bounded_all_formats():
    """All signal scores must be in [0,1] for JPEG and PNG inputs."""
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    rng = np.random.default_rng(99)
    arr = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    for fmt in ["JPEG", "PNG"]:
        det = AdvancedEnsembleDetector(_encode(arr, fmt), f"test.{fmt.lower()}")
        r   = det.detect()
        det.cleanup()
        for sig in r["all_signals"]:
            assert 0.0 <= sig["score"] <= 1.0, (
                f"{fmt}: signal {sig['signal_name']} score={sig['score']} out of range"
            )


def test_small_rotation_does_not_flip_verdict(sample_image_bytes):
    """Rotating image 5 degrees must not flip classification verdict."""
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    orig_img  = Image.open(BytesIO(sample_image_bytes)).convert("RGB")
    rot_img   = orig_img.rotate(5, expand=False)
    buf       = BytesIO()
    rot_img.save(buf, format="PNG")
    rot_bytes = buf.getvalue()
    det_orig = AdvancedEnsembleDetector(sample_image_bytes, "orig.png")
    r_orig   = det_orig.detect()
    det_orig.cleanup()
    det_rot  = AdvancedEnsembleDetector(rot_bytes, "rotated.png")
    r_rot    = det_rot.detect()
    det_rot.cleanup()
    classes  = ["likely_authentic", "possibly_authentic", "possibly_ai_generated", "likely_ai_generated"]
    orig_idx = classes.index(r_orig["classification"]) if r_orig["classification"] in classes else 2
    rot_idx  = classes.index(r_rot["classification"])  if r_rot["classification"]  in classes else 2
    assert abs(orig_idx - rot_idx) <= 1, (
        f"5-degree rotation flipped: {r_orig['classification']} -> {r_rot['classification']}"
    )


def test_mild_blur_score_stable(sample_image_bytes):
    """Mild blur must not change ai_probability by more than 0.25."""
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    from PIL import ImageFilter
    blurred = Image.open(BytesIO(sample_image_bytes)).convert("RGB").filter(
        ImageFilter.GaussianBlur(radius=1)
    )
    buf = BytesIO()
    blurred.save(buf, format="PNG")
    blurred_bytes = buf.getvalue()
    det_orig = AdvancedEnsembleDetector(sample_image_bytes, "orig.png")
    r_orig   = det_orig.detect()
    det_orig.cleanup()
    det_blur = AdvancedEnsembleDetector(blurred_bytes, "blurred.png")
    r_blur   = det_blur.detect()
    det_blur.cleanup()
    delta = abs(r_orig["ai_probability"] - r_blur["ai_probability"])
    assert delta < 0.25, f"Blur changed score by {delta:.3f}"


def test_ensemble_score_consistent_with_signals(sample_image_bytes):
    """Ensemble score must correlate with suspicious signal count."""
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    det = AdvancedEnsembleDetector(sample_image_bytes, "test.png")
    r   = det.detect()
    det.cleanup()
    suspicious = r["suspicious_signals_count"]
    prob       = r["ai_probability"]
    if suspicious > 16:
        assert prob > 0.5, f"{suspicious}/26 suspicious but prob={prob:.3f}"
    if suspicious < 6:
        assert prob < 0.65, f"Only {suspicious}/26 suspicious but prob={prob:.3f}"


def test_all_signals_have_required_schema(sample_image_bytes):
    """Every signal must have signal_name, score, confidence, explanation."""
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    det = AdvancedEnsembleDetector(sample_image_bytes, "test.png")
    r   = det.detect()
    det.cleanup()
    required = {"signal_name", "score", "confidence", "explanation"}
    for sig in r["all_signals"]:
        missing = required - set(sig.keys())
        assert not missing, f"Signal missing {missing}: {sig.get('signal_name')}"
        assert isinstance(sig["score"], float)
        assert isinstance(sig["confidence"], float)


def test_forensic_report_complete_schema(sample_image_bytes):
    """Full forensic report must always contain all top-level keys."""
    from backend.services.image_forensics import ImageForensics
    report = ImageForensics(sample_image_bytes, "test.png").generate_forensic_report()
    required = {
        "evidence_id", "metadata", "file_info", "exif_data",
        "hashes", "tampering_analysis", "ai_detection",
        "generator_attribution", "platform_forensics", "summary",
    }
    missing = required - set(report.keys())
    assert not missing, f"Report missing keys: {missing}"
    assert 0.0 <= report["summary"]["ai_probability"] <= 1.0
    assert report["summary"]["total_detection_signals"] == 26
    assert "platform_origin" in report["summary"]
