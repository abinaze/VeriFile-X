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


@pytest.fixture
def client():
    """Synchronous test client for API endpoint testing."""
    return TestClient(app)


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
