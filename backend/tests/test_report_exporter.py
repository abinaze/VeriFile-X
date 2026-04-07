"""
Tests for forensic report export suite (PDF, JSON, CSV).
"""
import json
import csv
import io
import pytest
import numpy as np
from PIL import Image
from io import BytesIO


def _make_image(seed: int = 42) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, (100, 100, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_mock_report() -> dict:
    """Minimal report dict for unit testing exporters."""
    return {
        "evidence_id": "test-evidence-uuid-1234",
        "metadata":    {"analysis_timestamp": "2026-04-07T12:00:00", "analyzer_version": "6.1.0"},
        "file_info":   {"filename": "test.jpg", "width": 100, "height": 100, "file_size_bytes": 5000},
        "hashes":      {"sha256": "a" * 64, "md5": "b" * 32, "perceptual_hash": "c0c0c0c0"},
        "exif_data":   {"has_exif": False},
        "tampering_analysis": {"suspicious_flags": [], "confidence": "high"},
        "ai_detection": {
            "ai_probability": 0.85,
            "classification": "likely_ai_generated",
            "suspicious_signals_count": 15,
            "total_signals": 26,
            "all_signals": [
                {
                    "signal_name": f"Signal {i}",
                    "score":       0.7 if i % 2 == 0 else 0.2,
                    "confidence":  0.8,
                    "explanation": f"Explanation for signal {i}",
                    "raw_value":   float(i),
                    "expected_range": "0-1",
                }
                for i in range(26)
            ],
            "methods_used": ["statistical", "clip"],
        },
        "generator_attribution": {"predicted_generator": "stylegan", "confidence": 0.72},
        "platform_forensics":    {"predicted_platform": "instagram", "confidence": 0.65},
        "c2pa_provenance":       {"provenance_status": "none", "has_c2pa": False},
        "summary": {
            "has_metadata": False,
            "suspicious_flags_count": 1,
            "authenticity_confidence": "medium",
            "ai_probability": 0.85,
            "ai_classification": "likely_ai_generated",
            "total_detection_signals": 26,
            "suspicious_detection_signals": 15,
            "predicted_generator": "stylegan",
            "platform_origin": "instagram",
            "c2pa_status": "none",
        },
    }


# ── JSON export ───────────────────────────────────────────────────────────────

def test_json_export_is_valid_json():
    from backend.services.report_exporter import export_json
    result = export_json(_make_mock_report())
    assert isinstance(result, bytes)
    parsed = json.loads(result)
    assert parsed["evidence_id"] == "test-evidence-uuid-1234"


def test_json_export_contains_all_keys():
    from backend.services.report_exporter import export_json
    report = _make_mock_report()
    result = json.loads(export_json(report))
    for key in ("evidence_id", "metadata", "file_info", "ai_detection", "summary"):
        assert key in result


# ── CSV export ────────────────────────────────────────────────────────────────

def test_csv_export_is_valid():
    from backend.services.report_exporter import export_csv
    result = export_csv(_make_mock_report())
    assert isinstance(result, bytes)
    rows = list(csv.reader(io.StringIO(result.decode("utf-8"))))
    assert len(rows) > 1  # header + data rows


def test_csv_export_header_correct():
    from backend.services.report_exporter import export_csv
    result = export_csv(_make_mock_report())
    reader = csv.reader(io.StringIO(result.decode("utf-8")))
    header = next(reader)
    assert "signal_name" in header
    assert "score" in header
    assert "evidence_id" in header


def test_csv_export_row_count():
    from backend.services.report_exporter import export_csv
    report = _make_mock_report()
    result = export_csv(report)
    rows   = list(csv.reader(io.StringIO(result.decode("utf-8"))))
    assert len(rows) == 27  # 1 header + 26 signals


def test_csv_export_scores_numeric():
    from backend.services.report_exporter import export_csv
    result = export_csv(_make_mock_report())
    reader = csv.DictReader(io.StringIO(result.decode("utf-8")))
    for row in reader:
        assert 0.0 <= float(row["score"]) <= 1.0
        assert 0.0 <= float(row["confidence"]) <= 1.0


# ── PDF export ────────────────────────────────────────────────────────────────

def test_pdf_export_returns_bytes():
    from backend.services.report_exporter import export_pdf
    result = export_pdf(_make_mock_report())
    assert isinstance(result, bytes)
    assert len(result) > 100


def test_pdf_export_starts_with_pdf_header():
    from backend.services.report_exporter import export_pdf
    result = export_pdf(_make_mock_report())
    assert result[:4] == b"%PDF"


def test_pdf_export_contains_eof():
    from backend.services.report_exporter import export_pdf
    result = export_pdf(_make_mock_report())
    assert b"%%EOF" in result


def test_pdf_export_handles_missing_fields():
    from backend.services.report_exporter import export_pdf
    result = export_pdf({})
    assert result[:4] == b"%PDF"


# ── API endpoint tests ────────────────────────────────────────────────────────

def test_export_json_endpoint(client):
    img = _make_image()
    response = client.post(
        "/api/v1/analyze/export/json",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert "evidence_id" in data


def test_export_csv_endpoint(client):
    img = _make_image(seed=11)
    response = client.post(
        "/api/v1/analyze/export/csv",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    rows = list(csv.reader(io.StringIO(response.text)))
    assert len(rows) > 1


def test_export_pdf_endpoint(client):
    img = _make_image(seed=22)
    response = client.post(
        "/api/v1/analyze/export/pdf",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.status_code == 200
    assert "application/pdf" in response.headers["content-type"]
    assert response.content[:4] == b"%PDF"


def test_export_invalid_format(client):
    img = _make_image()
    response = client.post(
        "/api/v1/analyze/export/xlsx",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.status_code == 400


def test_export_rejects_non_image(client):
    response = client.post(
        "/api/v1/analyze/export/json",
        files={"file": ("test.txt", b"text", "text/plain")}
    )
    assert response.status_code == 415


def test_export_content_disposition_header(client):
    img = _make_image(seed=33)
    for fmt in ("json", "csv", "pdf"):
        response = client.post(
            f"/api/v1/analyze/export/{fmt}",
            files={"file": ("report_test.jpg", img, "image/jpeg")}
        )
        assert response.status_code == 200
        assert "content-disposition" in response.headers
        assert "attachment" in response.headers["content-disposition"]
