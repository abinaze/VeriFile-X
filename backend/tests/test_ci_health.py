"""
CI health checks — fast smoke tests that verify the application
can be imported, configured, and started without errors.
These run first in CI to catch import-time failures immediately.
"""
import pytest


def test_app_imports_without_error():
    """FastAPI app must import cleanly."""
    from backend.main import app
    assert app is not None
    assert app.title == "VeriFile-X"


def test_all_routers_registered():
    """Verify major API route groups are registered (starlette-version-agnostic).

    Uses FastAPI's openapi() schema rather than walking app.routes directly —
    the starlette 1.x _IncludedRouter structure doesn't store the full path
    on sub-route objects, making recursive walks unreliable. The OpenAPI schema
    is the canonical source of all registered paths and works across versions.
    """
    from backend.main import app as _app

    registered = _app.openapi().get("paths", {})
    assert any(p.startswith("/api/v1/analyze") for p in registered), (
        f"analyze route not found. Registered paths: {sorted(registered.keys())}"
    )
    assert "/health" in registered or any(p == "/health" for p in registered), (
        f"health route not found. Registered paths: {sorted(registered.keys())}"
    )

def test_settings_loads():
    """Settings must load from environment without raising."""
    from backend.core.config import settings
    assert settings.PROJECT_NAME == "VeriFile-X"
    assert settings.MAX_FILE_SIZE_MB > 0
    assert settings.CACHE_TTL_MINUTES > 0


def test_logger_initializes():
    """Logger setup must not raise."""
    from backend.core.logger import setup_logger
    logger = setup_logger("test_module")
    assert logger is not None
    logger.info("CI health check logger test")


def test_cache_initializes():
    """ForensicsCache must initialize without error."""
    from backend.core.cache import ForensicsCache
    cache = ForensicsCache()
    assert cache is not None
    assert cache.size() == 0


def test_all_service_modules_importable():
    """All service modules must import without error."""
    services = [
        "backend.services.image_forensics",
        "backend.services.advanced_ensemble_detector",
        "backend.services.generator_attribution",
        "backend.services.platform_detector",
        "backend.services.c2pa_verifier",
        "backend.services.batch_processor",
        "backend.services.report_exporter",
        "backend.services.case_manager",
        "backend.services.api_key_manager",
        "backend.services.heatmap_generator",
        "backend.services.adversarial_tester",
    ]
    import importlib
    for svc in services:
        try:
            importlib.import_module(svc)
        except ImportError as e:
            pytest.fail(f"Service {svc} failed to import: {e}")


def test_api_routes_importable():
    """All API route modules must import without error."""
    import importlib
    for route in ["backend.api.routes.analyze",
                  "backend.api.routes.cases",
                  "backend.api.routes.keys",
                  "backend.api.routes.upload"]:
        importlib.import_module(route)


def test_health_endpoint_structure(client):
    """Health endpoint must return status, debug_mode, timestamp."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")  # degraded when model files absent (CI)
    assert "timestamp" in data
    assert "debug_mode" in data


def test_docs_endpoint_available(client):
    """OpenAPI docs must be accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_root_endpoint(client):
    """Root endpoint must respond 200."""
    response = client.get("/")
    assert response.status_code == 200


class TestSessionStartModelPrewarm:
    """conftest.py's pytest_sessionstart pre-warms the process-wide
    ModelCache (F-7) before any individual test's timeout clock starts,
    so the cold DIRE/CLIP/own-embedding load cost lands on session
    startup (no timeout) instead of on whichever full-pipeline test
    happens to run first alphabetically (previously inside the fast
    tier's 60s per-test budget -- see conftest.py's docstring for the
    full story, and PR #166's CI run for a real example of the failure
    this prevents: test_advanced_ai_detector.py::test_forensics_integration
    timed out at 60s despite doing nothing wrong itself)."""

    def test_sessionstart_calls_load_model_on_all_three_detectors(self, monkeypatch):
        """Direct regression test for the actual bug: pre-warming must
        touch DIRE, CLIP, and own-embedding, not just one or two of
        them -- any full-pipeline test can be the unlucky first one to
        touch whichever detector isn't pre-warmed."""
        import backend.tests.conftest as conftest_mod

        calls = []

        class _FakeDetector:
            def __init__(self, name):
                self._name = name
            def _load_model(self):
                calls.append(self._name)

        monkeypatch.setattr(
            "backend.services.dire_detector.DIREDetector",
            lambda: _FakeDetector("dire"),
        )
        monkeypatch.setattr(
            "backend.services.clip_detector.CLIPDetector",
            lambda: _FakeDetector("clip"),
        )
        monkeypatch.setattr(
            "backend.services.own_embedding_detector.OwnEmbeddingDetector",
            lambda: _FakeDetector("own"),
        )

        conftest_mod.pytest_sessionstart(session=None)

        assert set(calls) == {"dire", "clip", "own"}

    def test_sessionstart_never_raises_even_if_loading_fails(self, monkeypatch):
        """Pre-warming is an optimization, not a correctness requirement
        -- if it can't complete (no network, no torch, etc.), the
        session must still start normally and let each detector's own
        existing fallback-to-neutral-result handling take over
        per-test, exactly as it would have without this hook."""
        import backend.tests.conftest as conftest_mod

        def _raise():
            raise RuntimeError("simulated load failure")

        monkeypatch.setattr(
            "backend.services.dire_detector.DIREDetector",
            lambda: (_ for _ in ()).throw(RuntimeError("simulated import/construct failure")),
        )

        # Must not raise.
        conftest_mod.pytest_sessionstart(session=None)
