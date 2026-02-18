"""
Tests for in-memory forensics cache.
"""
import pytest
from backend.core.cache import ForensicsCache


@pytest.fixture
def cache():
    """Fresh cache instance for each test."""
    return ForensicsCache()


@pytest.fixture
def sample_report():
    """Sample forensic report for caching."""
    return {
        "file_info": {"filename": "test.png"},
        "summary": {"ai_probability": 0.3}
    }


def test_cache_miss_on_empty(cache, sample_image_bytes):
    """Empty cache returns None."""
    result = cache.get(sample_image_bytes)
    assert result is None


def test_cache_set_and_get(cache, sample_image_bytes, sample_report):
    """Store and retrieve report correctly."""
    cache.set(sample_image_bytes, sample_report)
    result = cache.get(sample_image_bytes)

    assert result is not None
    assert result["summary"]["ai_probability"] == 0.3


def test_cache_hit_adds_metadata(cache, sample_image_bytes, sample_report):
    """Cache hit includes cache_info metadata."""
    cache.set(sample_image_bytes, sample_report)
    result = cache.get(sample_image_bytes)

    assert "cache_info" in result
    assert result["cache_info"]["cached"] is True
    assert result["cache_info"]["cache_hit"] is True


def test_cache_different_files(cache, sample_image_bytes, sample_report):
    """Different files get different cache entries."""
    different_bytes = b"different_file_content"

    cache.set(sample_image_bytes, sample_report)

    assert cache.get(sample_image_bytes) is not None
    assert cache.get(different_bytes) is None


def test_cache_size(cache, sample_image_bytes, sample_report):
    """Cache size increments correctly."""
    assert cache.size() == 0
    cache.set(sample_image_bytes, sample_report)
    assert cache.size() == 1


def test_cache_clear(cache, sample_image_bytes, sample_report):
    """Cache clear removes all entries."""
    cache.set(sample_image_bytes, sample_report)
    cache.clear()
    assert cache.size() == 0
    assert cache.get(sample_image_bytes) is None
