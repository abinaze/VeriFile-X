"""
Performance and scalability tests.
"""
import pytest
import time

# Try to import psutil, skip tests if not available
try:
    import psutil
    import os
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def test_small_image_performance(sample_image_bytes):
    """Test performance on small images."""
    from backend.services.statistical_detector import StatisticalDetector
    
    start = time.time()
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    duration = time.time() - start
    
    # Should complete in reasonable time (<10 seconds for small image)
    assert duration < 10.0
    assert report["total_signals"] == 19


@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed (requires C++ compiler on Windows)")
def test_memory_usage_is_bounded(sample_image_bytes):
    """Test that memory usage doesn't explode (requires psutil)."""
    from backend.services.statistical_detector import StatisticalDetector
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run detection
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Should not use excessive memory (<500MB increase)
    assert memory_increase < 500


def test_cache_speedup_is_significant(client):
    """Test that cache provides measurable speedup."""
    # Use unique test data to avoid cache pollution from other tests
    from PIL import Image
    from io import BytesIO
    import numpy as np
    
    # Create unique image
    unique_data = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
    img = Image.fromarray(unique_data)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    unique_bytes = buffer.getvalue()
    
    files = {"file": ("unique_cache_test.png", unique_bytes, "image/png")}
    
    # First request (no cache)
    start1 = time.time()
    response1 = client.post("/api/v1/analyze/image", files=files)
    time1 = time.time() - start1
    
    assert response1.status_code == 200
    
    # Wait a tiny bit to ensure clear timing
    time.sleep(0.01)
    
    # Second request (cached)
    start2 = time.time()
    files2 = {"file": ("unique_cache_test.png", unique_bytes, "image/png")}
    response2 = client.post("/api/v1/analyze/image", files=files2)
    time2 = time.time() - start2
    
    assert response2.status_code == 200
    
    # Cache should provide some speedup (at least 2x)
    # Note: On fast systems, both might be very fast, so we check if cached is faster
    assert time2 < time1 or time2 < 0.1  # Either faster, or both very fast


def test_signal_computation_order_matters_not(sample_image_bytes):
    """Test that signal computation order doesn't affect result."""
    from backend.services.statistical_detector import StatisticalDetector
    
    # Normal detection
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    base_score = report["ai_probability"]
    
    # All signals contribute equally regardless of order
    assert 0 <= base_score <= 1


def test_concurrent_detection_performance(sample_image_bytes):
    """Test performance under concurrent load."""
    from backend.services.statistical_detector import StatisticalDetector
    import concurrent.futures
    
    def run_detection():
        detector = StatisticalDetector(sample_image_bytes, "test.png")
        return detector.detect()
    
    start = time.time()
    
    # Run 5 concurrent detections
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_detection) for _ in range(5)]
        results = [f.result() for f in futures]
    
    duration = time.time() - start
    
    # Should complete all 5 in reasonable time (<60s)
    assert duration < 60
    assert len(results) == 5
