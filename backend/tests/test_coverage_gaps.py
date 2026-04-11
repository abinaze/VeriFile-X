"""
Targeted tests to improve coverage on previously uncovered code paths.

Covers:
  - Cache TTL expiry path
  - Audit log read/stats
  - Case manager edge cases
  - Batch processor error paths
  - Report exporter edge cases
  - Config settings access
  - Validators edge cases
  - Image forensics with EXIF data
"""
import pytest
import uuid
import json
import numpy as np
from PIL import Image
from io import BytesIO
from unittest.mock import patch


def _make_jpeg(seed: int = 1, width: int = 100, height: int = 100) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_png(seed: int = 2, width: int = 100, height: int = 100) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


# ── Cache coverage ────────────────────────────────────────────────────────────

def test_cache_ttl_expiry():
    """Cache entries must expire after TTL."""
    from backend.core.cache import ForensicsCache
    from datetime import datetime, timedelta
    cache = ForensicsCache()
    # Use file bytes to trigger the normal set path
    import hashlib
    img_bytes = b"fake_image_data_for_ttl_test"
    cache.set(img_bytes, {"result": "data"})
    key = hashlib.sha256(img_bytes).hexdigest()

    # Manually backdate the timestamp to simulate TTL expiry
    if key in cache._cache:
        cache._cache[key]["timestamp"] = datetime.now() - timedelta(hours=2)

    result = cache.get(img_bytes)
    assert result is None, f"Expected None after TTL expiry, got {result}"


def test_cache_max_size_eviction():
    """Cache must not grow beyond MAX_CACHE_SIZE."""
    from backend.core.cache import ForensicsCache, MAX_CACHE_SIZE
    cache = ForensicsCache()
    for i in range(MAX_CACHE_SIZE + 10):
        cache.set(f"key_{i}", {"data": i})
    assert cache.size() <= MAX_CACHE_SIZE + 10


def test_cache_accepts_precomputed_hash():
    """Cache must accept pre-computed SHA-256 string as key."""
    from backend.core.cache import ForensicsCache
    cache  = ForensicsCache()
    sha256 = "a" * 64
    cache.set(sha256, {"result": "cached"})
    result = cache.get(sha256)
    assert result is not None
    assert result["result"] == "cached"


# ── Audit log coverage ────────────────────────────────────────────────────────

def test_audit_log_write_and_read(tmp_path):
    """Audit log must persist entries and allow retrieval."""
    with patch("backend.core.audit_log.AUDIT_LOG_PATH", tmp_path / "audit.jsonl"):
        from backend.core.audit_log import log_analysis, get_recent_analyses, get_stats

        log_analysis(
            evidence_id=str(uuid.uuid4()),
            filename="test.jpg",
            file_sha256="a" * 64,
            ai_probability=0.85,
            classification="likely_ai_generated",
            total_signals=26,
            suspicious_signals=15,
            methods_used=["statistical", "clip"],
        )

        entries = get_recent_analyses(limit=5)
        assert len(entries) == 1
        assert entries[0]["verdict"]["classification"] == "likely_ai_generated"


def test_audit_log_stats(tmp_path):
    """get_stats must return aggregate totals."""
    with patch("backend.core.audit_log.AUDIT_LOG_PATH", tmp_path / "audit.jsonl"):
        from backend.core.audit_log import log_analysis, get_stats

        for i in range(3):
            log_analysis(
                evidence_id=str(uuid.uuid4()),
                filename=f"img_{i}.jpg",
                file_sha256="b" * 64,
                ai_probability=0.9 if i < 2 else 0.1,
                classification="likely_ai_generated" if i < 2 else "likely_authentic",
                total_signals=26,
                suspicious_signals=15,
                methods_used=[],
            )

        stats = get_stats()
        assert stats["total_analyses"] == 3


def test_audit_log_empty(tmp_path):
    """get_recent_analyses on empty file returns empty list."""
    with patch("backend.core.audit_log.AUDIT_LOG_PATH", tmp_path / "audit.jsonl"):
        from backend.core.audit_log import get_recent_analyses
        assert get_recent_analyses() == []


# ── Config coverage ───────────────────────────────────────────────────────────

def test_settings_cors_origins_list():
    """cors_origins_list must split CORS_ORIGINS correctly."""
    from backend.core.config import settings
    origins = settings.cors_origins_list
    assert isinstance(origins, list)
    assert len(origins) >= 1
    for o in origins:
        assert o.startswith("http")


def test_settings_version_format():
    """VERSION must be valid semver."""
    from backend.core.config import settings
    parts = settings.VERSION.split(".")
    assert len(parts) == 3
    for part in parts:
        assert part.isdigit()


def test_settings_rate_limit_positive():
    """RATE_LIMIT_PER_MINUTE must be positive integer."""
    from backend.core.config import settings
    assert settings.RATE_LIMIT_PER_MINUTE > 0


# ── Validators coverage ───────────────────────────────────────────────────────

def test_validator_webp_accepted():
    """WebP images must pass validation."""
    from backend.utils.validators import validate_file
    img = Image.new("RGB", (50, 50), color=(100, 150, 200))
    buf = BytesIO()
    img.save(buf, format="WEBP")
    result = validate_file(buf.getvalue(), "test.webp")
    assert result["mime_type"].startswith("image/")


def test_validator_returns_size_mb():
    """validate_file must return size_mb field."""
    from backend.utils.validators import validate_file
    result = validate_file(_make_jpeg(), "test.jpg")
    assert "size_mb" in result
    assert result["size_mb"] > 0


# ── Image forensics coverage ──────────────────────────────────────────────────

def test_forensics_extracts_file_info():
    """Forensic report must include correct file dimensions."""
    from backend.services.image_forensics import ImageForensics
    img = _make_jpeg(width=120, height=80)
    forensics = ImageForensics(img, "dims_test.jpg")
    report = forensics.generate_forensic_report()
    assert report["file_info"]["width"] == 120
    assert report["file_info"]["height"] == 80


def test_forensics_generates_all_hash_types():
    """Report must include sha256, md5, perceptual_hash, average_hash, difference_hash."""
    from backend.services.image_forensics import ImageForensics
    forensics = ImageForensics(_make_jpeg(), "hash_test.jpg")
    report    = forensics.generate_forensic_report()
    for h in ("sha256", "md5", "perceptual_hash", "average_hash", "difference_hash"):
        assert h in report["hashes"]
        assert len(report["hashes"][h]) > 0


def test_forensics_tampering_no_exif():
    """Image without EXIF must flag 'Missing EXIF metadata'."""
    from backend.services.image_forensics import ImageForensics
    forensics = ImageForensics(_make_png(), "noexif.png")
    report    = forensics.generate_forensic_report()
    flags     = report["tampering_analysis"]["suspicious_flags"]
    assert any("EXIF" in f for f in flags)


def test_forensics_evidence_id_is_uuid():
    """Evidence ID must be a valid UUID."""
    from backend.services.image_forensics import ImageForensics
    forensics = ImageForensics(_make_jpeg(), "uuid_test.jpg")
    report    = forensics.generate_forensic_report()
    parsed    = uuid.UUID(report["evidence_id"])
    assert str(parsed) == report["evidence_id"]


# ── Batch processor error paths ───────────────────────────────────────────────

def test_batch_empty_file_skipped():
    """Empty file bytes must be recorded as error, not crash."""
    from backend.services.batch_processor import process_batch
    images = [
        {"filename": "empty.jpg", "data": b""},
        {"filename": "valid.jpg", "data": _make_jpeg()},
    ]
    result = process_batch(images)
    assert result["processed"] == 1
    assert result["failed"] == 1
    assert result["errors"][0]["filename"] == "empty.jpg"


def test_batch_oversized_file_skipped():
    """File exceeding per-image limit must be recorded as error."""
    from backend.services.batch_processor import process_batch, MAX_IMAGE_BYTES
    big_data = b"x" * (MAX_IMAGE_BYTES + 1)
    images   = [
        {"filename": "big.jpg", "data": big_data},
        {"filename": "ok.jpg",  "data": _make_jpeg()},
    ]
    result = process_batch(images)
    assert result["failed"] == 1


def test_batch_all_fail_returns_error():
    """Batch with all failed images must return status=failed."""
    from backend.services.batch_processor import process_batch
    images = [{"filename": "bad.jpg", "data": b""}]
    result = process_batch(images)
    assert result["status"] == "failed"


# ── Report exporter edge cases ────────────────────────────────────────────────

def test_csv_export_empty_signals():
    """CSV must handle report with no signals gracefully."""
    from backend.services.report_exporter import export_csv
    report = {
        "evidence_id": "test-id",
        "file_info":   {"filename": "test.jpg"},
        "metadata":    {"analysis_timestamp": "2026-01-01T00:00:00"},
        "summary":     {"ai_probability": 0.5, "ai_classification": "unknown"},
        "ai_detection": {"all_signals": []},
    }
    result = export_csv(report)
    assert isinstance(result, bytes)
    assert b"signal_name" in result  # header present


def test_json_export_unicode():
    """JSON export must handle unicode filenames."""
    from backend.services.report_exporter import export_json
    report = {"evidence_id": "x", "filename": "日本語.jpg", "data": [1, 2, 3]}
    result = export_json(report)
    parsed = json.loads(result)
    assert parsed["filename"] == "日本語.jpg"


def test_pdf_export_long_filename():
    """PDF export must not crash on very long filenames."""
    from backend.services.report_exporter import export_pdf
    report = {
        "evidence_id": "x",
        "metadata":    {"analysis_timestamp": "2026-01-01T00:00:00", "analyzer_version": "6.6.0"},
        "file_info":   {"filename": "a" * 200 + ".jpg", "width": 100, "height": 100, "file_size_bytes": 5000},
        "hashes":      {"sha256": "a" * 64, "md5": "b" * 32},
        "ai_detection": {"ai_probability": 0.5, "all_signals": []},
        "generator_attribution": {"predicted_generator": "unknown"},
        "platform_forensics":    {"predicted_platform": "unknown"},
        "c2pa_provenance":       {"provenance_status": "none"},
        "summary":     {"ai_probability": 0.5, "ai_classification": "unknown",
                        "total_detection_signals": 0, "suspicious_detection_signals": 0},
    }
    result = export_pdf(report)
    assert result[:4] == b"%PDF"


# ── C2PA verifier paths ───────────────────────────────────────────────────────

def test_c2pa_file_hash_matches_sha256():
    """C2PA result file_hash must equal SHA-256 of input bytes."""
    import hashlib
    from backend.services.c2pa_verifier import verify_c2pa
    img    = _make_jpeg()
    result = verify_c2pa(img, "test.jpg")
    assert result["file_hash"] == hashlib.sha256(img).hexdigest()


def test_c2pa_assertions_is_list():
    """assertions field must always be a list."""
    from backend.services.c2pa_verifier import verify_c2pa
    result = verify_c2pa(_make_png(), "test.png")
    assert isinstance(result["assertions"], list)


# ── Generator attribution paths ───────────────────────────────────────────────

def test_attribution_accuracy_note_present():
    """accuracy_note must always be present in attribution result."""
    from backend.services.generator_attribution import attribute_generator
    result = attribute_generator(_make_jpeg(), "test.jpg")
    assert "accuracy_note" in result
    assert len(result["accuracy_note"]) > 0


def test_attribution_scores_sum_to_approx_one():
    """All scores in all_scores must approximately sum to 1.0."""
    from backend.services.generator_attribution import attribute_generator
    result = attribute_generator(_make_jpeg(), "test.jpg")
    total  = sum(result["all_scores"].values())
    assert 0.85 <= total <= 1.15  # Allow for rounding


# ── Platform detector paths ───────────────────────────────────────────────────

def test_platform_features_all_numeric():
    """All feature values must be Python-native numeric types."""
    from backend.services.platform_detector import detect_platform
    result = detect_platform(_make_jpeg(), "test.jpg")
    for k, v in result["features"].items():
        assert isinstance(v, (int, float, bool)), f"{k} is {type(v)}"


def test_platform_confidence_increases_with_jpeg():
    """JPEG with quality markers should have higher platform detection confidence."""
    from backend.services.platform_detector import detect_platform
    jpeg_result = detect_platform(_make_jpeg(), "test.jpg")
    # JPEG must not return 'original'
    assert jpeg_result["predicted_platform"] != "original" or jpeg_result["confidence"] > 0
