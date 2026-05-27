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
    """All expected routers must be registered on the app."""
    from backend.main import app
    prefixes = {route.path for route in app.routes}
    # Check that key endpoints exist
    paths_str = " ".join(str(p) for p in prefixes)
    assert "/api/v1/analyze/image" in paths_str or "/api/v1/analyze" in paths_str
    assert "/health" in paths_str


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
