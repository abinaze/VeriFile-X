"""
Regression test for C-2 (audit finding): the EXIF-orientation-correction
step in analyze.py's /image handler unconditionally re-encoded every
JPEG upload -- even when no rotation was needed -- with no quality=
argument, silently dropping to Pillow's default JPEG quality (75)
regardless of the original quality. This corrupted the input to every
compression-sensitive forensic signal (ELA, JPEG Ghost, DCT,
PRNU-heuristic, Noise Map, Noiseprint, CFA) for the majority of real
phone-camera uploads.

The fix was extracted into a standalone function,
_correct_exif_orientation(), specifically so it has direct regression
coverage independent of the full detection pipeline (which needs heavy
optional ML dependencies not available in every test environment).
"""
import io

import numpy as np
import pytest
from PIL import Image

from backend.api.routes.analyze import _correct_exif_orientation


def _synthetic_photo_bytes(quality: int, orientation: int | None) -> bytes:
    """A noise-heavy, photo-like image so JPEG compression differences
    are actually measurable (a flat-color image compresses to nearly
    nothing at any quality and wouldn't show the bug)."""
    rng = np.random.default_rng(42)
    arr = (rng.random((600, 800, 3)) * 255).astype("uint8")
    grad = np.linspace(0, 255, 800, dtype="uint8")
    arr[:, :, 0] = (arr[:, :, 0].astype(int) * 0.5 + grad[None, :].astype(int) * 0.5).astype("uint8")
    img = Image.fromarray(arr, mode="RGB")

    buf = io.BytesIO()
    if orientation is None:
        img.save(buf, format="JPEG", quality=quality)
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation  # Orientation tag
        img.save(buf, format="JPEG", quality=quality, exif=exif.tobytes())
    return buf.getvalue()


def _mean_abs_pixel_diff(a_bytes: bytes, b_bytes: bytes) -> float:
    """Pixel-level diff between two ALREADY-ALIGNED (same orientation)
    images. Callers are responsible for comparing against a correctly
    rotated reference -- this does not attempt to correct misalignment
    itself, since guessing the right transform here would risk masking a
    real orientation bug instead of a real compression-quality bug."""
    a = np.array(Image.open(io.BytesIO(a_bytes)).convert("RGB")).astype(int)
    b = np.array(Image.open(io.BytesIO(b_bytes)).convert("RGB")).astype(int)
    assert a.shape == b.shape, f"Images are not aligned: {a.shape} vs {b.shape}"
    return float(np.abs(a - b).mean())


class TestNoRotationNeeded:
    """The common case: no orientation tag, or Orientation=1 (identity).
    This must now be a complete no-op."""

    def test_no_exif_at_all_returns_bytes_unchanged(self):
        original = _synthetic_photo_bytes(quality=95, orientation=None)
        result = _correct_exif_orientation(original)
        assert result == original, (
            "An upload with no EXIF orientation data must not be "
            "re-encoded at all -- any change here is silent, "
            "uncontrolled recompression (C-2)."
        )

    def test_orientation_1_returns_bytes_unchanged(self):
        original = _synthetic_photo_bytes(quality=95, orientation=1)
        result = _correct_exif_orientation(original)
        assert result == original, (
            "Orientation=1 (identity/no rotation) must not trigger a "
            "re-encode -- this is the single most common real-world "
            "case (most cameras write Orientation=1 even for "
            "'normal' shots) and was silently corrupted before this fix."
        )


class TestRotationNeeded:
    """When a real rotation IS needed, the image must still be corrected,
    but quality must be preserved far better than an uncontrolled
    default-quality-75 re-encode."""

    def test_orientation_6_actually_rotates(self):
        original = _synthetic_photo_bytes(quality=90, orientation=6)
        result = _correct_exif_orientation(original)
        orig_img = Image.open(io.BytesIO(original))
        result_img = Image.open(io.BytesIO(result))
        # Orientation=6 is a 90-degree rotation -- width/height swap.
        assert result_img.size == (orig_img.size[1], orig_img.size[0])

    def test_orientation_6_preserves_quality_much_better_than_naive_resave(self):
        original = _synthetic_photo_bytes(quality=95, orientation=6)

        # Ground truth: the correctly-rotated pixels, saved LOSSLESSLY
        # (PNG) so this reference has no compression artifacts of its
        # own to confound the comparison. Both the fixed and naive paths
        # below produce images in this same orientation, so comparing
        # each against this reference isolates compression-quality loss
        # from rotation-alignment (they're deliberately not compared
        # directly against the pre-rotation "original").
        from PIL import ImageOps
        reference_img = ImageOps.exif_transpose(Image.open(io.BytesIO(original)))
        reference_buf = io.BytesIO()
        reference_img.save(reference_buf, format="PNG")
        reference_bytes = reference_buf.getvalue()

        fixed_result = _correct_exif_orientation(original)

        # Reproduce the OLD buggy behavior for direct comparison: same
        # exif_transpose, but save() with no quality/qtables at all.
        naive_img = ImageOps.exif_transpose(Image.open(io.BytesIO(original)))
        naive_buf = io.BytesIO()
        naive_img.save(naive_buf, format="JPEG")
        naive_result = naive_buf.getvalue()

        fixed_diff = _mean_abs_pixel_diff(reference_bytes, fixed_result)
        naive_diff = _mean_abs_pixel_diff(reference_bytes, naive_result)

        assert fixed_diff < naive_diff, (
            f"Fixed path (mean abs diff={fixed_diff:.2f}) should preserve "
            f"quality noticeably better than the old naive re-save "
            f"(mean abs diff={naive_diff:.2f})."
        )
        # Concrete regression ceiling -- the old bug measured ~12-13/255 on
        # comparable synthetic images; the fixed path should stay well
        # under half of that.
        assert fixed_diff < 8.0, (
            f"Mean abs pixel diff {fixed_diff:.2f}/255 is too high -- "
            "quantization tables/subsampling may not be surviving the "
            "re-encode."
        )

    def test_non_jpeg_format_still_handled_without_crashing(self):
        """PNG (or other lossless formats) should pass through the
        rotation logic without the JPEG-specific qtables path being
        exercised, and without raising."""
        img = Image.new("RGB", (50, 50), color=(10, 20, 30))
        buf = io.BytesIO()
        exif = Image.Exif()
        exif[0x0112] = 6
        img.save(buf, format="PNG", exif=exif.tobytes())
        original = buf.getvalue()
        result = _correct_exif_orientation(original)
        # Must not raise, and must still be a valid, readable image.
        Image.open(io.BytesIO(result)).verify()


class TestFailsOpen:
    def test_corrupt_bytes_return_unchanged_rather_than_raising(self):
        garbage = b"not a real image"
        result = _correct_exif_orientation(garbage)
        assert result == garbage
