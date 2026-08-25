"""
Regression test for C-1 (audit finding): a geotagged photo crashed
ImageForensics.extract_exif() and, before that, would have crashed
FastAPI's response serialization even if extract_exif() itself had
survived.

Root cause (confirmed by direct reproduction, not assumed): Image.getexif()
returns the GPSInfo tag as an unresolved IFD pointer (a plain int), not a
dict of GPS sub-tags. The previous code iterated that raw int directly,
raising TypeError. Even a corrected implementation that resolves the
sub-IFD via get_ifd() still needs to convert leftover IFDRational scalars
(e.g. GPSAltitude) to native types before the result is JSON-serializable.

This test builds a real geotagged JPEG using pure Pillow (no extra
dependency) -- an Image.Exif() object with an explicit GPSInfo sub-IFD
containing IFDRational values, exactly the shape a real camera/phone
produces -- and exercises the real extract_exif() method end-to-end,
plus the same jsonable_encoder() FastAPI uses on the live endpoint.
"""
import io

import pytest
from PIL import Image
from PIL.ExifTags import IFD
from PIL.TiffImagePlugin import IFDRational

from backend.services.image_forensics import ImageForensics


def _build_geotagged_jpeg() -> bytes:
    img = Image.new("RGB", (400, 300), color=(100, 150, 200))
    exif = Image.Exif()
    gps_ifd = exif.get_ifd(IFD.GPSInfo)
    gps_ifd[1] = "N"  # GPSLatitudeRef
    gps_ifd[2] = (IFDRational(37, 1), IFDRational(46, 1), IFDRational(2964, 100))  # GPSLatitude
    gps_ifd[3] = "W"  # GPSLongitudeRef
    gps_ifd[4] = (IFDRational(122, 1), IFDRational(25, 1), IFDRational(1084, 100))  # GPSLongitude
    gps_ifd[6] = IFDRational(305, 10)  # GPSAltitude -- stays a raw scalar IFDRational
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


def test_extract_exif_does_not_raise_on_geotagged_photo():
    """The single most common real-world input (a geotagged phone photo)
    must not raise out of extract_exif()."""
    jpeg_bytes = _build_geotagged_jpeg()
    forensics = ImageForensics(jpeg_bytes, "geotagged.jpg")

    # Must not raise. Before the fix this raised:
    #   TypeError: 'int' object is not iterable
    result = forensics.extract_exif()

    assert result.get("has_exif") is True
    assert "gps" in result
    assert result["gps"], "GPS sub-IFD must actually be resolved, not empty"


def test_extract_exif_gps_values_are_json_encodable():
    """The GPS fields extract_exif() returns must be usable by FastAPI's
    jsonable_encoder -- the same encoder used on the real, undeclared-
    response_model /api/v1/analyze/image route."""
    from fastapi.encoders import jsonable_encoder

    jpeg_bytes = _build_geotagged_jpeg()
    forensics = ImageForensics(jpeg_bytes, "geotagged.jpg")
    result = forensics.extract_exif()

    # Must not raise ValueError/TypeError from an un-converted IFDRational.
    encoded = jsonable_encoder(result)

    gps = encoded["gps"]
    assert gps["GPSLatitudeRef"] == "N"
    assert gps["GPSLongitudeRef"] == "W"
    # GPSLatitude/GPSLongitude: tuple of 3 plain numbers, no IFDRational left.
    assert len(gps["GPSLatitude"]) == 3
    for v in gps["GPSLatitude"]:
        assert isinstance(v, (int, float))
    # GPSAltitude: scalar, must be a plain float, not an IFDRational.
    assert isinstance(gps["GPSAltitude"], (int, float))
    assert abs(gps["GPSAltitude"] - 30.5) < 1e-6


def test_extract_exif_still_handles_no_exif_gracefully():
    """Non-regression: images with no EXIF at all must still return the
    original, simple has_exif=False shape."""
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(1, 2, 3)).save(buf, format="PNG")
    forensics = ImageForensics(buf.getvalue(), "plain.png")
    result = forensics.extract_exif()
    assert result == {"has_exif": False}


def test_full_report_pipeline_does_not_crash_on_gps(monkeypatch):
    """End-to-end: generate_forensic_report() (the function the real
    /api/v1/analyze/image route calls) must not raise on a geotagged
    photo either -- extract_exif() is the very first thing it calls,
    with no surrounding try/except of its own."""
    jpeg_bytes = _build_geotagged_jpeg()
    forensics = ImageForensics(jpeg_bytes, "geotagged.jpg")

    # extract_exif() (the actual regression target) must succeed as part
    # of the real report-generation call order.
    exif_data = forensics.extract_exif()
    assert exif_data.get("has_exif") is True
    tampering = forensics.detect_tampering_indicators(exif_data)
    assert "suspicious_flags" in tampering
