"""
Tests for Phase 6-9 detection signals:
PRNU, ELA, Metadata Forensics, DCT Frequency.
"""
import pytest
import numpy as np
from io import BytesIO
from PIL import Image


@pytest.fixture
def sample_image_bytes():
    np.random.seed(42)
    arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return buf.getvalue()


def test_prnu_detector_returns_valid_signal(sample_image_bytes):
    from backend.services.prnu_detector import detect_prnu
    result = detect_prnu(sample_image_bytes, "test.png")
    assert "signal_name" in result
    assert "score" in result
    assert "confidence" in result
    assert 0.0 <= result["score"] <= 1.0
    assert result["signal_name"] == "PRNU Camera Fingerprint"


def test_ela_detector_returns_valid_signal(sample_image_bytes):
    from backend.services.ela_detector import detect_ela
    result = detect_ela(sample_image_bytes, "test.png")
    assert "signal_name" in result
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0
    assert result["signal_name"] == "ELA Compression Analysis"


def test_metadata_forensics_no_exif(sample_image_bytes):
    from backend.services.metadata_forensics import analyze_metadata
    result = analyze_metadata(sample_image_bytes, "test.png")
    assert "signal_name" in result
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0
    assert result["signal_name"] == "Metadata Forensics"
    # PNG with no EXIF should flag missing metadata
    assert result["score"] > 0.5


def test_dct_detector_returns_valid_signal(sample_image_bytes):
    from backend.services.dct_frequency_detector import detect_dct_artifacts
    result = detect_dct_artifacts(sample_image_bytes, "test.png")
    assert "signal_name" in result
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0
    assert result["signal_name"] == "DCT Frequency Artifacts"


def test_all_new_signals_have_explanation(sample_image_bytes):
    from backend.services.prnu_detector import detect_prnu
    from backend.services.ela_detector import detect_ela
    from backend.services.metadata_forensics import analyze_metadata
    from backend.services.dct_frequency_detector import detect_dct_artifacts

    for fn, name in [
        (detect_prnu, "PRNU Camera Fingerprint"),
        (detect_ela, "ELA Compression Analysis"),
        (analyze_metadata, "Metadata Forensics"),
        (detect_dct_artifacts, "DCT Frequency Artifacts"),
    ]:
        result = fn(sample_image_bytes, "test.png")
        assert "explanation" in result, f"{name} missing explanation"
        assert len(result["explanation"]) > 10, f"{name} explanation too short"


def test_signals_handle_tiny_image():
    """All signals should gracefully handle tiny images."""
    from backend.services.prnu_detector import detect_prnu
    from backend.services.ela_detector import detect_ela
    from backend.services.dct_frequency_detector import detect_dct_artifacts

    tiny = BytesIO()
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(tiny, format="PNG")
    tiny_bytes = tiny.getvalue()

    for fn in [detect_prnu, detect_ela, detect_dct_artifacts]:
        result = fn(tiny_bytes, "tiny.png")
        assert 0.0 <= result["score"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0  # Graceful handling for tiny images


def test_signals_handle_corrupt_data():
    """All signals should return fallback on corrupt input."""
    from backend.services.prnu_detector import detect_prnu
    from backend.services.ela_detector import detect_ela
    from backend.services.metadata_forensics import analyze_metadata
    from backend.services.dct_frequency_detector import detect_dct_artifacts

    bad = b"not an image at all"
    for fn in [detect_prnu, detect_ela, analyze_metadata, detect_dct_artifacts]:
        result = fn(bad, "corrupt.png")
        assert "score" in result
        assert result["score"] == 0.5  # Neutral fallback
        assert result["confidence"] == 0.0
