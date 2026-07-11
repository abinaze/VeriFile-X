"""
Shared test fixtures.
Why: Reusable, realistic test data across all test files.
"""
import pytest
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app


def pytest_configure(config):
    """
    Register custom pytest markers.
    
    Markers:
    - slow: Tests that take >2 seconds (ML model loading, large computations)
    - integration: Tests that require multiple components
    - gpu: Tests that require CUDA/GPU
    """
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require GPU/CUDA"
    )


@pytest.fixture(autouse=True)
def _test_api_key(tmp_path, monkeypatch):
    """
    Auth was added to analyze.py/cases.py (see backend/core/auth.py).
    Every existing test that hits those routers via the `client` fixture
    needs a valid analyst-role key — this creates one against an isolated,
    per-test keys file so tests never touch the real data/api_keys.jsonl.
    """
    from backend.services import api_key_manager
    temp_keys = tmp_path / "test_api_keys.jsonl"
    monkeypatch.setattr(api_key_manager, "KEYS_PATH", temp_keys)
    result = api_key_manager.create_key("pytest-analyst", role="analyst")
    return result["key"]


@pytest.fixture
def client(_test_api_key):
    """
    Synchronous test client for API endpoint testing.
    Auto-attaches a valid analyst Authorization header to every request,
    since analyze.py/cases.py now require auth (backend/core/auth.py).
    Tests that specifically need to verify auth-rejection should
    construct their own bare TestClient(app) instead of using this fixture.
    """
    test_client = TestClient(app)
    test_client.headers.update({"Authorization": f"Bearer {_test_api_key}"})
    return test_client


@pytest.fixture
def sample_image_bytes():
    """
    Generate a realistic 100x100 test image.

    Why 100x100 with noise?
    - 1x1 pixel → all statistical metrics = 0 (meaningless)
    - Gradient + Gaussian noise → simulates real camera photo
    - Provides meaningful data for:
        * Laplacian variance (noise analysis)
        * FFT (frequency domain)
        * 8x8 block analysis (JPEG artifacts)
        * Color entropy (HSV histogram)
    """
    np.random.seed(42)  # Deterministic for reproducible tests

    # Build 100x100 RGB image with gradient base
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)

    for i in range(100):
        for j in range(100):
            img_array[i, j] = [
                int(i * 2.5),           # R: vertical gradient
                int(j * 2.5),           # G: horizontal gradient
                int((i + j) * 1.25)     # B: diagonal gradient
            ]

    # Add Gaussian noise (simulates camera sensor noise)
    # Real photos: Noise ~ N(0, σ²), σ ≈ 10-20 for typical cameras
    noise = np.random.normal(0, 15, img_array.shape).astype(np.int16)
    img_array = np.clip(
        img_array.astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)

    # Encode as PNG bytes
    buffer = BytesIO()
    Image.fromarray(img_array, 'RGB').save(buffer, format='PNG')
    buffer.seek(0)

    return buffer.read()
