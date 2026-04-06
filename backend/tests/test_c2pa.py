"""
Tests for C2PA content credentials verification.
All tests must pass without any external C2PA SDK installed.
"""
import pytest
import numpy as np
from PIL import Image
from io import BytesIO

_VALID_STATUSES = {"verified", "partial", "none", "tampered", "unknown"}


def _make_jpeg(width: int = 128, height: int = 128) -> bytes:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_png(width: int = 128, height: int = 128) -> bytes:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _inject_xmp_c2pa(image_bytes: bytes) -> bytes:
    """Inject a fake C2PA XMP marker into image bytes for testing."""
    xmp = (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description rdf:about="" xmlns:dcterms="http://purl.org/dc/terms/">'
        b'<dcterms:provenance>c2pa.manifest:test-manifest-id</dcterms:provenance>'
        b'<stds-org:c2pa xmlns:stds-org="http://ns.useplus.org/">1.0</stds-org:c2pa>'
        b'</rdf:Description></rdf:RDF></x:xmpmeta>'
        b'<?xpacket end="w"?>'
    )
    return image_bytes + xmp


def test_c2pa_returns_valid_status():
    from backend.services.c2pa_verifier import verify_c2pa
    result = verify_c2pa(_make_jpeg(), "test.jpg")
    assert result["provenance_status"] in _VALID_STATUSES


def test_c2pa_no_manifest_returns_none():
    """Plain image with no C2PA should return status 'none'."""
    from backend.services.c2pa_verifier import verify_c2pa
    result = verify_c2pa(_make_jpeg(), "test.jpg")
    assert result["provenance_status"] == "none"
    assert result["has_c2pa"] is False


def test_c2pa_confidence_in_range():
    from backend.services.c2pa_verifier import verify_c2pa
    result = verify_c2pa(_make_jpeg(), "test.jpg")
    assert 0.0 <= result["confidence"] <= 1.0


def test_c2pa_required_fields_present():
    from backend.services.c2pa_verifier import verify_c2pa
    result = verify_c2pa(_make_jpeg(), "test.jpg")
    required = {
        "provenance_status", "has_c2pa", "manifest_found",
        "signing_info", "ai_training_policy", "assertions",
        "soft_binding_valid", "confidence", "explanation",
        "file_hash", "accuracy_note",
    }
    missing = required - set(result.keys())
    assert not missing, f"Missing fields: {missing}"


def test_c2pa_file_hash_is_sha256():
    from backend.services.c2pa_verifier import verify_c2pa
    import hashlib
    img = _make_jpeg()
    result = verify_c2pa(img, "test.jpg")
    expected = hashlib.sha256(img).hexdigest()
    assert result["file_hash"] == expected


def test_c2pa_xmp_marker_detected():
    """Image with injected XMP C2PA marker must return has_c2pa=True."""
    from backend.services.c2pa_verifier import verify_c2pa
    img_with_c2pa = _inject_xmp_c2pa(_make_jpeg())
    result = verify_c2pa(img_with_c2pa, "c2pa_test.jpg")
    assert result["has_c2pa"] is True
    assert result["provenance_status"] in {"verified", "partial"}
    assert len(result["assertions"]) > 0


def test_c2pa_ai_training_policy_structure():
    from backend.services.c2pa_verifier import verify_c2pa
    result = verify_c2pa(_make_jpeg(), "test.jpg")
    policy = result["ai_training_policy"]
    assert "ai_training_allowed" in policy
    assert "has_explicit_policy" in policy
    assert "policy_note" in policy
    assert isinstance(policy["ai_training_allowed"], bool)


def test_c2pa_handles_corrupt_data():
    from backend.services.c2pa_verifier import verify_c2pa
    result = verify_c2pa(b"not_an_image", "corrupt.bin")
    assert result["provenance_status"] in _VALID_STATUSES


def test_c2pa_api_endpoint(client):
    img = _make_jpeg()
    response = client.post(
        "/api/v1/analyze/c2pa",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "provenance_status" in data
    assert "has_c2pa" in data
    assert "confidence" in data


def test_c2pa_api_rejects_non_image(client):
    response = client.post(
        "/api/v1/analyze/c2pa",
        files={"file": ("test.txt", b"text", "text/plain")}
    )
    assert response.status_code == 415


def test_c2pa_in_forensic_report(client):
    """Main analysis endpoint must include c2pa_provenance in response."""
    rng = np.random.default_rng(seed=7777)
    arr = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("c2pa_unique.png", buf.getvalue(), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "c2pa_provenance" in data
    assert data["c2pa_provenance"]["provenance_status"] in _VALID_STATUSES
    assert "c2pa_status" in data["summary"]


def test_c2pa_png_returns_none():
    """Standard PNG without C2PA should return none."""
    from backend.services.c2pa_verifier import verify_c2pa
    result = verify_c2pa(_make_png(), "test.png")
    assert result["provenance_status"] == "none"
    assert result["has_c2pa"] is False
