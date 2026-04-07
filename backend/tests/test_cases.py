"""
Tests for evidence case management system.
Uses a temporary cases.jsonl for isolation.
"""
import pytest
import uuid
from pathlib import Path
from unittest.mock import patch


# Patch CASES_PATH to a temp file for all tests
@pytest.fixture(autouse=True)
def temp_cases_file(tmp_path):
    temp = tmp_path / "test_cases.jsonl"
    with patch("backend.services.case_manager.CASES_PATH", temp):
        yield temp


# ── Service unit tests ────────────────────────────────────────────────────────

def test_create_case_returns_case_id():
    from backend.services.case_manager import create_case
    result = create_case("Test Case", "A test investigation")
    assert "case_id" in result
    assert result["name"] == "Test Case"
    assert result["status"] == "open"
    assert result["evidence"] == []


def test_create_case_requires_name():
    from backend.services.case_manager import create_case
    result = create_case("")
    assert "error" in result


def test_create_case_persists():
    from backend.services.case_manager import create_case, get_case
    case = create_case("Persistent Case")
    retrieved = get_case(case["case_id"])
    assert retrieved["case_id"] == case["case_id"]
    assert retrieved["name"] == "Persistent Case"


def test_add_evidence_to_case():
    from backend.services.case_manager import create_case, add_evidence
    case = create_case("Evidence Case")
    result = add_evidence(
        case_id=case["case_id"],
        evidence_id=str(uuid.uuid4()),
        filename="test.jpg",
        ai_probability=0.85,
        classification="likely_ai_generated",
        notes="Suspicious image from social media",
    )
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["ai_probability"] == 0.85
    assert result["evidence"][0]["classification"] == "likely_ai_generated"


def test_add_evidence_to_nonexistent_case():
    from backend.services.case_manager import add_evidence
    result = add_evidence(
        case_id=str(uuid.uuid4()),
        evidence_id=str(uuid.uuid4()),
        filename="test.jpg",
        ai_probability=0.5,
        classification="unknown",
    )
    assert "error" in result


def test_add_evidence_to_closed_case():
    from backend.services.case_manager import create_case, add_evidence, update_status
    case = create_case("Closed Case")
    update_status(case["case_id"], "closed")
    result = add_evidence(
        case_id=case["case_id"],
        evidence_id=str(uuid.uuid4()),
        filename="test.jpg",
        ai_probability=0.5,
        classification="unknown",
    )
    assert "error" in result


def test_list_cases_returns_all():
    from backend.services.case_manager import create_case, list_cases
    create_case("Case A")
    create_case("Case B")
    cases = list_cases()
    assert len(cases) == 2


def test_list_cases_filter_by_status():
    from backend.services.case_manager import create_case, list_cases, update_status
    c1 = create_case("Open Case")
    c2 = create_case("Closed Case")
    update_status(c2["case_id"], "closed")
    open_cases = list_cases(status="open")
    assert len(open_cases) == 1
    assert open_cases[0]["case_id"] == c1["case_id"]


def test_update_status_valid():
    from backend.services.case_manager import create_case, update_status
    case = create_case("Status Test")
    result = update_status(case["case_id"], "closed")
    assert result["status"] == "closed"


def test_update_status_invalid():
    from backend.services.case_manager import create_case, update_status
    case = create_case("Status Test")
    result = update_status(case["case_id"], "invalid_status")
    assert "error" in result


def test_search_cases_by_name():
    from backend.services.case_manager import create_case, search_cases
    create_case("Election Fraud Investigation", tags=["election", "2024"])
    create_case("Unrelated Case")
    results = search_cases("election")
    assert len(results) == 1
    assert "Election" in results[0]["name"]


def test_search_cases_by_tag():
    from backend.services.case_manager import create_case, search_cases
    create_case("Tagged Case", tags=["deepfake", "politics"])
    results = search_cases("deepfake")
    assert len(results) == 1


def test_delete_case_archives():
    from backend.services.case_manager import create_case, delete_case, get_case
    case = create_case("To Delete")
    delete_case(case["case_id"])
    retrieved = get_case(case["case_id"])
    assert retrieved["status"] == "archived"


def test_case_summary_stats():
    from backend.services.case_manager import create_case, add_evidence, get_case_summary
    case = create_case("Summary Test")
    for i, prob in enumerate([0.8, 0.9, 0.2]):
        add_evidence(
            case_id=case["case_id"],
            evidence_id=str(uuid.uuid4()),
            filename=f"img_{i}.jpg",
            ai_probability=prob,
            classification="likely_ai_generated" if prob > 0.5 else "likely_authentic",
        )
    summary = get_case_summary(case["case_id"])
    assert summary["evidence_count"] == 3
    assert summary["ai_detected"] == 2
    assert 0.0 <= summary["mean_ai_prob"] <= 1.0


# ── API endpoint tests ────────────────────────────────────────────────────────

def test_api_create_case(client):
    response = client.post(
        "/api/v1/cases/",
        json={"name": "API Test Case", "description": "Created via API", "tags": ["test"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "case_id" in data
    assert data["name"] == "API Test Case"


def test_api_list_cases(client):
    client.post("/api/v1/cases/", json={"name": "List Test"})
    response = client.get("/api/v1/cases/")
    assert response.status_code == 200
    assert "cases" in response.json()


def test_api_get_case(client):
    create_resp = client.post("/api/v1/cases/", json={"name": "Get Test"})
    case_id = create_resp.json()["case_id"]
    response = client.get(f"/api/v1/cases/{case_id}")
    assert response.status_code == 200
    assert response.json()["case_id"] == case_id


def test_api_get_nonexistent_case(client):
    response = client.get(f"/api/v1/cases/{uuid.uuid4()}")
    assert response.status_code == 404


def test_api_add_evidence(client):
    case_resp = client.post("/api/v1/cases/", json={"name": "Evidence Test"})
    case_id   = case_resp.json()["case_id"]
    response  = client.post(
        f"/api/v1/cases/{case_id}/evidence",
        json={
            "evidence_id":    str(uuid.uuid4()),
            "filename":       "suspect.jpg",
            "ai_probability": 0.92,
            "classification": "likely_ai_generated",
            "notes":          "Found on social media",
            "tags":           ["social_media"],
        }
    )
    assert response.status_code == 200
    assert len(response.json()["evidence"]) == 1


def test_api_update_status(client):
    case_resp = client.post("/api/v1/cases/", json={"name": "Status Test"})
    case_id   = case_resp.json()["case_id"]
    response  = client.patch(
        f"/api/v1/cases/{case_id}/status",
        json={"status": "closed"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "closed"


def test_api_delete_archives(client):
    case_resp = client.post("/api/v1/cases/", json={"name": "Delete Test"})
    case_id   = case_resp.json()["case_id"]
    response  = client.delete(f"/api/v1/cases/{case_id}")
    assert response.status_code == 200
    get_resp  = client.get(f"/api/v1/cases/{case_id}")
    assert get_resp.json()["status"] == "archived"


def test_api_search(client):
    client.post("/api/v1/cases/", json={"name": "Deepfake Investigation 2024"})
    client.post("/api/v1/cases/", json={"name": "Unrelated"})
    response = client.get("/api/v1/cases/search?q=deepfake")
    assert response.status_code == 200
    cases = response.json()["cases"]
    assert len(cases) == 1
    assert "Deepfake" in cases[0]["name"]
