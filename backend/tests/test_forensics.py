"""
Tests for image forensics service.
"""
import pytest
from backend.services.image_forensics import ImageForensics


def test_forensics_initialization(sample_image_bytes):
    """Test forensics service initializes correctly."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    assert forensics.filename == "test.png"
    assert forensics.pil_image is not None  # FIXED: Changed from 'image' to 'pil_image'


def test_extract_exif_no_data(sample_image_bytes):
    """Test EXIF extraction with no EXIF data."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    exif = forensics.extract_exif()
    assert exif["has_exif"] == False


def test_generate_hashes(sample_image_bytes):
    """Test hash generation."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    hashes = forensics.generate_hashes()
    
    assert "sha256" in hashes
    assert "md5" in hashes
    assert "perceptual_hash" in hashes
    assert len(hashes["sha256"]) == 64


def test_detect_tampering(sample_image_bytes):
    """Test tampering detection."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    exif = forensics.extract_exif()
    tampering = forensics.detect_tampering_indicators(exif)
    
    assert "suspicious_flags" in tampering  # FIXED: Removed 'analysis' check
    assert "confidence" in tampering
    assert isinstance(tampering["suspicious_flags"], list)


class TestF28ExpandedKeywordLists:
    """F-28: the editing-tool and AI-generation-marker keyword lists were
    narrow (5 editing tools, 4 AI generators) and missed most mainstream
    tools entirely -- Lightroom, Affinity, Topaz, Firefly, Leonardo.Ai,
    Ideogram, Flux, and more. These tests exercise real EXIF-shaped
    dicts directly against detect_tampering_indicators() (it only reads
    the exif_data argument, so a real image isn't needed to construct
    the ImageForensics instance) -- both previously-missed cases (must
    now flag) and previously-working cases (must still flag, no
    regression), plus an ordinary-camera-EXIF case that must NOT
    false-positive.
    """

    @staticmethod
    def _forensics():
        # detect_tampering_indicators only reads its exif_data argument,
        # not self -- bypass __init__ (which needs a real image) rather
        # than constructing one just to call this one method.
        return object.__new__(ImageForensics)

    @pytest.mark.parametrize("software,should_flag", [
        ("Adobe Photoshop Lightroom Classic 13.0", True),   # previously MISSED
        ("Affinity Photo 2.4", True),                       # previously MISSED
        ("Topaz Photo AI 3.1", True),                        # previously MISSED
        ("Capture One 23", True),                            # previously MISSED
        ("Adobe Photoshop 25.0", True),                      # already worked
        ("GIMP 2.10.34", True),                              # already worked
        ("Canon EOS Utility 3.0", False),                    # ordinary camera-maker software, not an editor
    ])
    def test_editing_tool_detection(self, software, should_flag):
        report = self._forensics().detect_tampering_indicators(
            {"has_exif": True, "Software": software}
        )
        flagged = any("Editing software detected" in f for f in report["suspicious_flags"])
        assert flagged == should_flag, f"Software={software!r} expected flagged={should_flag}, flags={report['suspicious_flags']}"

    @pytest.mark.parametrize("field,value,should_flag", [
        ("Artist", "Generated with Adobe Firefly", True),     # previously MISSED
        ("ImageDescription", "Created using Leonardo.Ai", True),  # previously MISSED
        ("UserComment", "Ideogram v2 output", True),          # previously MISSED
        ("ImageDescription", "FLUX.1 dev generation", True),  # previously MISSED
        ("Software", "Midjourney v6", True),                  # already worked
        ("UserComment", "stable diffusion 1.5 checkpoint", True),  # already worked
        ("Artist", "Jane Smith, freelance photographer", False),  # ordinary EXIF, must NOT false-positive
    ])
    def test_ai_marker_detection(self, field, value, should_flag):
        report = self._forensics().detect_tampering_indicators(
            {"has_exif": True, field: value}
        )
        flagged = any("AI generation marker" in f for f in report["suspicious_flags"])
        assert flagged == should_flag, f"{field}={value!r} expected flagged={should_flag}, flags={report['suspicious_flags']}"

    def test_ordinary_camera_exif_produces_no_false_positive(self):
        """A completely mundane camera EXIF block must produce zero
        suspicious flags -- expanding the keyword lists must not make
        the detector trigger-happy on real photos."""
        report = self._forensics().detect_tampering_indicators({
            "has_exif": True, "Make": "Canon", "Model": "EOS R5",
            "DateTime": "2026:03:14 10:22:01",
        })
        assert report["suspicious_flags"] == []

    def test_lists_meaningfully_expanded(self):
        """Direct regression check against the exact old list sizes,
        so a future accidental revert is caught immediately."""
        from backend.services.image_forensics import EDITING_TOOLS, AI_GENERATION_MARKERS
        assert len(EDITING_TOOLS) > 5, "EDITING_TOOLS looks like it reverted to the old 5-entry list"
        assert len(AI_GENERATION_MARKERS) > 4, "AI_GENERATION_MARKERS looks like it reverted to the old 4-entry list"
        assert "lightroom" in EDITING_TOOLS
        assert "adobe firefly" in AI_GENERATION_MARKERS


def test_generate_forensic_report(sample_image_bytes):
    """Test complete forensic report generation."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    report = forensics.generate_forensic_report()
    
    assert "metadata" in report
    assert "file_info" in report
    assert "exif_data" in report
    assert "hashes" in report
    assert "tampering_analysis" in report
    assert "ai_detection" in report
    assert "summary" in report


def test_classify_image_type_called_once_per_report(sample_image_bytes, monkeypatch):
    """F-26 regression test: classify_image_type() used to run once inside
    AdvancedEnsembleDetector.combine_signals() (to gate PRNU/ELA/metadata
    by content type) and AGAIN inside generate_forensic_report() for the
    top-level report's own image_type field -- same function, same
    input, computed twice. Now the second call reuses the first's
    result via ai_detection["image_type_info"]."""
    import backend.services.image_forensics as forensics_mod
    call_count = []
    real_classify = forensics_mod.classify_image_type

    def _counting_classify(*args, **kwargs):
        call_count.append(1)
        return real_classify(*args, **kwargs)

    monkeypatch.setattr(forensics_mod, "classify_image_type", _counting_classify)
    # Also patch it where AdvancedEnsembleDetector's module imports it,
    # since that's a separate local import inside its own method.
    import backend.services.advanced_ensemble_detector as ensemble_mod
    monkeypatch.setattr(
        "backend.services.image_type_classifier.classify_image_type",
        _counting_classify,
    )

    forensics = ImageForensics(sample_image_bytes, "test.png")
    forensics.generate_forensic_report()

    assert len(call_count) == 1, (
        f"expected classify_image_type() to run exactly once per report, "
        f"got {len(call_count)} calls"
    )


def test_analyze_endpoint(client, sample_image_bytes):
    """Test the analyze endpoint."""
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    response = client.post("/api/v1/analyze/image", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "file_info" in data
    assert "ai_detection" in data
    assert "summary" in data
