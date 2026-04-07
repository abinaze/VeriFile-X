"""
Tests for batch investigation mode.
"""
import pytest
import numpy as np
from PIL import Image
from io import BytesIO


def _make_jpeg(seed: int = 42, width: int = 100, height: int = 100) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, (height, width, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_batch_images(n: int = 3) -> list:
    return [{"filename": f"img_{i}.jpg", "data": _make_jpeg(seed=i*10)}
            for i in range(n)]


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_batch_statistics_structure():
    from backend.services.batch_processor import _batch_statistics
    mock_reports = [
        {"summary": {"ai_probability": 0.8, "ai_classification": "likely_ai_generated"},
         "generator_attribution": {"predicted_generator": "stylegan"},
         "c2pa_provenance": {"provenance_status": "none"},
         "file_info": {"filename": "a.jpg"}},
        {"summary": {"ai_probability": 0.2, "ai_classification": "likely_authentic"},
         "generator_attribution": {"predicted_generator": "real"},
         "c2pa_provenance": {"provenance_status": "none"},
         "file_info": {"filename": "b.jpg"}},
    ]
    stats = _batch_statistics(mock_reports)
    assert stats["total_images"] == 2
    assert stats["ai_detected_count"] == 1
    assert stats["authentic_count"] == 1
    assert 0.0 <= stats["mean_ai_probability"] <= 1.0
    assert "batch_verdict" in stats
    assert "classification_breakdown" in stats


def test_risk_ranking_is_descending():
    from backend.services.batch_processor import _rank_by_risk
    mock_reports = [
        {"summary": {"ai_probability": 0.3, "ai_classification": "likely_authentic"},
         "file_info": {"filename": "low.jpg"},
         "evidence_id": "aaa",
         "generator_attribution": {"predicted_generator": "real"},
         "c2pa_provenance": {"provenance_status": "none"}},
        {"summary": {"ai_probability": 0.9, "ai_classification": "likely_ai_generated"},
         "file_info": {"filename": "high.jpg"},
         "evidence_id": "bbb",
         "generator_attribution": {"predicted_generator": "stylegan"},
         "c2pa_provenance": {"provenance_status": "none"}},
    ]
    ranked = _rank_by_risk(mock_reports)
    assert ranked[0]["filename"] == "high.jpg"
    assert ranked[1]["filename"] == "low.jpg"
    assert ranked[0]["ai_probability"] >= ranked[1]["ai_probability"]


def test_duplicate_detection_identical():
    from backend.services.batch_processor import _find_duplicates
    # Same hash = distance 0 = identical
    mock_reports = [
        {"hashes": {"perceptual_hash": "f0f0f0f0f0f0f0f0"},
         "file_info": {"filename": "a.jpg"}},
        {"hashes": {"perceptual_hash": "f0f0f0f0f0f0f0f0"},
         "file_info": {"filename": "b.jpg"}},
    ]
    pairs = _find_duplicates(mock_reports)
    assert len(pairs) == 1
    assert pairs[0]["similarity"] == "identical"
    assert pairs[0]["phash_distance"] == 0


def test_duplicate_detection_no_match():
    from backend.services.batch_processor import _find_duplicates
    mock_reports = [
        {"hashes": {"perceptual_hash": "0000000000000000"},
         "file_info": {"filename": "a.jpg"}},
        {"hashes": {"perceptual_hash": "ffffffffffffffff"},
         "file_info": {"filename": "b.jpg"}},
    ]
    pairs = _find_duplicates(mock_reports)
    assert len(pairs) == 0


def test_provenance_consistency_no_credentials():
    from backend.services.batch_processor import _provenance_consistency
    reports = [
        {"c2pa_provenance": {"provenance_status": "none"}},
        {"c2pa_provenance": {"provenance_status": "none"}},
    ]
    result = _provenance_consistency(reports)
    assert result["consistency"] == "consistent_no_credentials"
    assert result["images_with_credentials"] == 0


def test_provenance_consistency_mixed():
    from backend.services.batch_processor import _provenance_consistency
    reports = [
        {"c2pa_provenance": {"provenance_status": "verified"}},
        {"c2pa_provenance": {"provenance_status": "none"}},
    ]
    result = _provenance_consistency(reports)
    assert result["consistency"] == "inconsistent"


def test_batch_rejects_oversized():
    from backend.services.batch_processor import process_batch, MAX_BATCH_SIZE
    images = [{"filename": f"img_{i}.jpg", "data": b"x"} for i in range(MAX_BATCH_SIZE + 1)]
    result = process_batch(images)
    assert "error" in result
    assert result["received"] > MAX_BATCH_SIZE


def test_batch_processes_multiple_images():
    from backend.services.batch_processor import process_batch
    images = _make_batch_images(2)
    result = process_batch(images)
    assert result["status"] == "complete"
    assert result["processed"] == 2
    assert "statistics" in result
    assert "risk_ranking" in result
    assert "duplicate_pairs" in result
    assert "provenance_consistency" in result
    assert len(result["individual_reports"]) == 2


def test_batch_statistics_values_bounded():
    from backend.services.batch_processor import process_batch
    result = process_batch(_make_batch_images(2))
    stats = result["statistics"]
    assert 0.0 <= stats["mean_ai_probability"] <= 1.0
    assert 0.0 <= stats["max_ai_probability"] <= 1.0
    assert stats["total_images"] == 2
    assert stats["batch_verdict"] in {"high_risk", "mixed", "likely_authentic"}


def test_batch_individual_reports_have_evidence_id():
    from backend.services.batch_processor import process_batch
    result = process_batch(_make_batch_images(2))
    for item in result["individual_reports"]:
        assert "evidence_id" in item
        assert item["status"] == "success"


def test_batch_api_endpoint(client):
    imgs = [_make_jpeg(seed=i) for i in range(3)]
    files = [("files", (f"img_{i}.jpg", img, "image/jpeg"))
             for i, img in enumerate(imgs)]
    response = client.post("/api/v1/analyze/batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "statistics" in data
    assert "risk_ranking" in data
    assert data["processed"] == 3


def test_batch_api_rejects_too_many(client):
    from backend.services.batch_processor import MAX_BATCH_SIZE
    imgs  = [_make_jpeg(seed=i) for i in range(MAX_BATCH_SIZE + 1)]
    files = [("files", (f"img_{i}.jpg", img, "image/jpeg"))
             for i, img in enumerate(imgs)]
    response = client.post("/api/v1/analyze/batch", files=files)
    assert response.status_code == 413


def test_batch_api_rejects_non_image(client):
    files = [("files", ("test.txt", b"text", "text/plain"))]
    response = client.post("/api/v1/analyze/batch", files=files)
    # Accept 415 (unsupported type) or 429 (rate limit hit in CI after other batch tests)
    assert response.status_code in (415, 429)
