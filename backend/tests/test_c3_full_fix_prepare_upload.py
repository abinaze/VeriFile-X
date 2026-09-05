"""
Regression tests for C-3's full fix (audit finding): before this, only
POST /image ran all six upload guards (content-type check, pre-read
Content-Length guard, EXIF-orientation correction, post-read size check,
MIME/extension validation via validate_file(), and the image-quality
gate). The other 8 file-accepting endpoints in this router had an
inconsistent subset -- the C-3 stopgap (test_c3_content_length_precheck.py)
already fixed the Content-Length half; this covers the remaining
consolidation into the shared _prepare_image_bytes()/prepare_upload()
pipeline in analyze.py.

Three things this file specifically exists to prove:
  1. Every migrated endpoint now actually enforces MIME/extension
     validation and the quality gate (previously it silently didn't, for
     8 of the 9 endpoints) -- proven by sending an intentionally-tiny
     image and expecting 422 where the endpoint would previously have
     returned 200.
  2. /platform and /c2pa specifically do NOT get EXIF-orientation
     correction, even though they now share the same pipeline as their
     siblings -- both fingerprint properties of the original file
     structure (EXIF presence/absence; a binary JUMBF C2PA manifest)
     that a re-encode would corrupt or destroy (see analyze.py's
     _prepare_image_bytes() docstring for the full rationale).
  3. /segment and /batch -- the two endpoints that don't go through
     prepare_upload() unchanged (/segment previously had no content-type
     check at all; /batch calls _prepare_image_bytes() directly per file
     for Content-Length-precheck reasons) -- behave correctly under the
     new pipeline too.
"""
import io

import numpy as np
import pytest
from PIL import Image


def _make_jpeg(width: int = 128, height: int = 128, quality: int = 85, seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_tiny_jpeg() -> bytes:
    """Below MIN_WIDTH/MIN_HEIGHT (64x64) -- the quality gate must reject
    this with 422 on every endpoint that runs it."""
    return _make_jpeg(width=32, height=32)


def _make_rotated_jpeg(quality: int = 90) -> bytes:
    """A real photo-like JPEG with EXIF Orientation=6 (90-degree
    rotation) -- the shape a real sideways phone photo has."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation
    img.save(buf, format="JPEG", quality=quality, exif=exif.tobytes())
    return buf.getvalue()


# ── Unit-level: _prepare_image_bytes() ──────────────────────────────────────

class TestPrepareImageBytesUnit:
    def test_correct_exif_true_rotates_oriented_image(self):
        from backend.api.routes.analyze import _prepare_image_bytes
        original = _make_rotated_jpeg()
        orig_w, orig_h = Image.open(io.BytesIO(original)).size
        prepared, _cap = _prepare_image_bytes(
            original, "photo.jpg", max_bytes=10_000_000, correct_exif=True
        )
        new_w, new_h = Image.open(io.BytesIO(prepared)).size
        assert (new_w, new_h) == (orig_h, orig_w), (
            "correct_exif=True must actually rotate an Orientation=6 image "
            "(width/height swap) -- see C-2 for the full rationale."
        )

    def test_correct_exif_false_leaves_oriented_image_untouched(self):
        from backend.api.routes.analyze import _prepare_image_bytes
        original = _make_rotated_jpeg()
        orig_w, orig_h = Image.open(io.BytesIO(original)).size
        prepared, _cap = _prepare_image_bytes(
            original, "photo.jpg", max_bytes=10_000_000, correct_exif=False
        )
        new_w, new_h = Image.open(io.BytesIO(prepared)).size
        assert (new_w, new_h) == (orig_w, orig_h), (
            "correct_exif=False must leave an Orientation=6 image's pixel "
            "dimensions exactly as uploaded -- no rotation, no re-encode."
        )

    def test_oversized_after_correction_raises_413(self):
        from fastapi import HTTPException
        from backend.api.routes.analyze import _prepare_image_bytes
        data = _make_jpeg(width=64, height=64)
        with pytest.raises(HTTPException) as exc_info:
            _prepare_image_bytes(data, "big.jpg", max_bytes=10, correct_exif=True)
        assert exc_info.value.status_code == 413

    def test_non_image_bytes_raise_422(self):
        from fastapi import HTTPException
        from backend.api.routes.analyze import _prepare_image_bytes
        with pytest.raises(HTTPException) as exc_info:
            _prepare_image_bytes(b"not an image at all", "junk.bin", max_bytes=10_000_000)
        assert exc_info.value.status_code == 422

    def test_tiny_image_raises_422_via_quality_gate(self):
        from fastapi import HTTPException
        from backend.api.routes.analyze import _prepare_image_bytes
        with pytest.raises(HTTPException) as exc_info:
            _prepare_image_bytes(_make_tiny_jpeg(), "tiny.jpg", max_bytes=10_000_000)
        assert exc_info.value.status_code == 422
        assert "unsuitable" in exc_info.value.detail.lower()

    def test_valid_image_returns_bytes_and_confidence_cap(self):
        from backend.api.routes.analyze import _prepare_image_bytes
        data = _make_jpeg(width=300, height=300)
        prepared, cap = _prepare_image_bytes(data, "ok.jpg", max_bytes=10_000_000)
        assert isinstance(prepared, (bytes, bytearray))
        assert 0.0 <= cap <= 1.0

    def test_empty_filename_defaults_without_crashing(self):
        """Several endpoints used to pass `file.filename or "upload"` ad
        hoc; the shared function now does this once, centrally."""
        from backend.api.routes.analyze import _prepare_image_bytes
        data = _make_jpeg(width=300, height=300)
        prepared, _cap = _prepare_image_bytes(data, "", max_bytes=10_000_000)
        assert isinstance(prepared, (bytes, bytearray))
        prepared, _cap = _prepare_image_bytes(data, None, max_bytes=10_000_000)
        assert isinstance(prepared, (bytes, bytearray))


# ── Endpoints that previously had NO quality gate now reject tiny images ────

class TestMigratedEndpointsEnforceQualityGate:
    """Before C-3's full fix, only /image ran the quality gate -- a
    32x32 image would previously have gone straight through to each of
    these endpoints' own service function (all of which handle small
    images gracefully at the function level, per their own unit tests --
    see test_attribution.py::test_attribution_handles_small_image etc.).
    It must now be rejected at the router level, consistently, before
    ever reaching those functions."""

    def test_heatmap_rejects_tiny_image(self, client):
        response = client.post(
            "/api/v1/analyze/image/heatmap",
            files={"file": ("tiny.jpg", _make_tiny_jpeg(), "image/jpeg")},
        )
        assert response.status_code == 422

    def test_attribution_rejects_tiny_image(self, client):
        response = client.post(
            "/api/v1/analyze/attribution",
            files={"file": ("tiny.jpg", _make_tiny_jpeg(), "image/jpeg")},
        )
        assert response.status_code == 422

    def test_platform_rejects_tiny_image(self, client):
        response = client.post(
            "/api/v1/analyze/platform",
            files={"file": ("tiny.jpg", _make_tiny_jpeg(), "image/jpeg")},
        )
        assert response.status_code == 422

    def test_c2pa_rejects_tiny_image(self, client):
        response = client.post(
            "/api/v1/analyze/c2pa",
            files={"file": ("tiny.jpg", _make_tiny_jpeg(), "image/jpeg")},
        )
        assert response.status_code == 422

    def test_robustness_rejects_tiny_image(self, client):
        response = client.post(
            "/api/v1/analyze/robustness",
            files={"file": ("tiny.jpg", _make_tiny_jpeg(), "image/jpeg")},
        )
        assert response.status_code == 422

    def test_export_rejects_tiny_image(self, client):
        response = client.post(
            "/api/v1/analyze/export/json",
            files={"file": ("tiny.jpg", _make_tiny_jpeg(), "image/jpeg")},
        )
        assert response.status_code == 422

    def test_stream_rejects_tiny_image(self, client):
        response = client.post(
            "/api/v1/analyze/image/stream",
            files={"file": ("tiny.jpg", _make_tiny_jpeg(), "image/jpeg")},
        )
        assert response.status_code == 422

    def test_segment_rejects_tiny_image(self, client):
        response = client.post(
            "/api/v1/analyze/segment",
            files={"file": ("tiny.jpg", _make_tiny_jpeg(), "image/jpeg")},
        )
        assert response.status_code == 422


# ── /platform and /c2pa must never rotate, even under the shared pipeline ──

class TestPlatformAndC2paSkipExifCorrection:
    """The one load-bearing, non-obvious design constraint this whole
    consolidation depends on: /platform keys partly on EXIF
    presence/absence, and /c2pa looks for a binary JUMBF provenance
    manifest -- both properties a generic EXIF-correcting re-encode would
    corrupt or destroy. Patches the SOURCE module
    (backend.services.platform_detector / backend.services.c2pa_verifier),
    not analyze.py's own namespace -- analyze.py imports these functions
    locally, inside the route handler, on every call, so the attribute
    that needs patching lives on the source module, not on
    backend.api.routes.analyze itself.
    """

    def test_platform_does_not_rotate_oriented_image(self, client, monkeypatch):
        captured = {}

        def _fake_detect_platform(file_bytes, filename):
            captured["size"] = Image.open(io.BytesIO(file_bytes)).size
            return {
                "predicted_platform": "original",
                "confidence": 0.9,
                "all_scores": {
                    "whatsapp": 0.0, "instagram": 0.0, "discord": 0.0,
                    "telegram": 0.0, "twitter_x": 0.0, "facebook": 0.0,
                    "original": 0.9,
                },
                "features": {"estimated_quality": 90, "has_exif": True, "hf_ratio": 0.1},
            }

        monkeypatch.setattr(
            "backend.services.platform_detector.detect_platform",
            _fake_detect_platform,
        )

        original = _make_rotated_jpeg()
        orig_size = Image.open(io.BytesIO(original)).size
        response = client.post(
            "/api/v1/analyze/platform",
            files={"file": ("rotated.jpg", original, "image/jpeg")},
        )
        assert response.status_code == 200
        assert captured["size"] == orig_size, (
            "detect_platform() must receive the image at its ORIGINAL "
            "orientation -- correct_exif must be False for this endpoint."
        )

    def test_c2pa_does_not_rotate_oriented_image(self, client, monkeypatch):
        captured = {}

        def _fake_verify_c2pa(file_bytes, filename):
            captured["size"] = Image.open(io.BytesIO(file_bytes)).size
            return {
                "provenance_status": "none", "has_c2pa": False,
                "manifest_found": False, "signing_info": {},
                "ai_training_policy": {
                    "ai_training_allowed": False,
                    "has_explicit_policy": False,
                    "policy_note": "no manifest",
                },
                "assertions": [], "soft_binding_valid": False,
                "confidence": 0.5, "explanation": "no manifest found",
                "file_hash": "0" * 64, "accuracy_note": "heuristic",
            }

        monkeypatch.setattr(
            "backend.services.c2pa_verifier.verify_c2pa",
            _fake_verify_c2pa,
        )

        original = _make_rotated_jpeg()
        orig_size = Image.open(io.BytesIO(original)).size
        response = client.post(
            "/api/v1/analyze/c2pa",
            files={"file": ("rotated.jpg", original, "image/jpeg")},
        )
        assert response.status_code == 200
        assert captured["size"] == orig_size, (
            "verify_c2pa() must receive the image at its ORIGINAL "
            "orientation -- correct_exif must be False for this endpoint."
        )

    def test_attribution_DOES_rotate_oriented_image_for_contrast(self, client, monkeypatch):
        """Sanity check/contrast case: a sibling endpoint that SHOULD
        correct -- proves the difference above is deliberate per-endpoint
        configuration, not an accident of how the fake functions behave."""
        captured = {}

        def _fake_attribute_generator(file_bytes, filename):
            captured["size"] = Image.open(io.BytesIO(file_bytes)).size
            return {
                "predicted_generator": "real", "confidence": 0.5,
                "all_scores": {g: 0.0 for g in (
                    "stylegan", "dalle3", "sd14", "sdxl", "midjourney",
                    "gpt4o", "flux", "imagen3", "ideogram", "recraft", "real",
                )},
                "method": "rule_based",
                "features": {"mean_hf": 0.0, "checker_ratio": 0.0, "noise_std": 0.0},
            }

        monkeypatch.setattr(
            "backend.services.generator_attribution.attribute_generator",
            _fake_attribute_generator,
        )

        original = _make_rotated_jpeg()
        orig_w, orig_h = Image.open(io.BytesIO(original)).size
        response = client.post(
            "/api/v1/analyze/attribution",
            files={"file": ("rotated.jpg", original, "image/jpeg")},
        )
        assert response.status_code == 200
        assert captured["size"] == (orig_h, orig_w), (
            "attribute_generator() must receive the ROTATED image -- "
            "/attribution uses correct_exif=True."
        )


# ── /segment: newly gains a content-type header check ──────────────────────

class TestSegmentGainsContentTypeCheck:
    """Real, previously-undocumented finding from this consolidation:
    /segment was the only one of this router's 9 file-accepting
    endpoints with no content-type header check at all -- real image
    bytes with a mismatched declared content-type would previously have
    been accepted (validate_file() sniffs actual content, not the
    header). It now 415s consistently with every sibling endpoint."""

    def test_segment_rejects_mismatched_content_type_header(self, client):
        real_jpeg = _make_jpeg(width=200, height=200)
        response = client.post(
            "/api/v1/analyze/segment",
            files={"file": ("test.jpg", real_jpeg, "text/plain")},
        )
        assert response.status_code == 415

    def test_segment_accepts_valid_image(self, client):
        real_jpeg = _make_jpeg(width=200, height=200)
        response = client.post(
            "/api/v1/analyze/segment",
            files={"file": ("test.jpg", real_jpeg, "image/jpeg")},
        )
        assert response.status_code == 200


# ── /batch: per-file guards via _prepare_image_bytes(), skip-and-continue ──

class TestBatchGainsPerFileGuards:
    def test_batch_skips_tiny_image_but_processes_valid_ones(self, client):
        # /batch is rate-limited to 2/minute per client, and slowapi's
        # in-memory storage is shared for the whole pytest session (see
        # test_hardening.py's identical reset for the same reason) --
        # without this, this test flakes with 429 depending on how many
        # other tests already called /batch earlier in the same run.
        import backend.api.routes.analyze as analyze_mod
        analyze_mod.limiter.limiter.storage.reset()

        good1 = _make_jpeg(width=150, height=150, seed=1)
        good2 = _make_jpeg(width=150, height=150, seed=2)
        tiny = _make_tiny_jpeg()
        files = [
            ("files", ("good1.jpg", good1, "image/jpeg")),
            ("files", ("tiny.jpg", tiny, "image/jpeg")),
            ("files", ("good2.jpg", good2, "image/jpeg")),
        ]
        response = client.post("/api/v1/analyze/batch", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["processed"] == 2, (
            "The quality gate must reject the tiny image before it's "
            "added to the batch, while the two valid images still process "
            "normally (skip-and-continue, not fail-the-whole-batch)."
        )
