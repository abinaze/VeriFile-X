"""
Tests for model cache system.
"""
import pytest
import time
from backend.core.model_cache import ModelCache, get_model_cache


def test_cache_singleton():
    """Test that ModelCache is a singleton."""
    cache1 = ModelCache()
    cache2 = ModelCache()
    cache3 = get_model_cache()
    
    assert cache1 is cache2
    assert cache2 is cache3
    
    # All should reference the same instance
    cache1.clear()  # Clear first
    cache1.set('test', 'value', 1.0)
    assert cache2.get('test') == 'value'
    assert cache3.get('test') == 'value'
    cache1.clear()  # Clean up


def test_cache_basic_operations():
    """Test basic cache get/set operations."""
    cache = ModelCache()
    cache.clear()
    cache.reset_stats()
    
    # Cache miss
    assert cache.get('model1') is None
    
    # Cache set
    cache.set('model1', {'data': 'test'}, 100.0)
    
    # Cache hit
    result = cache.get('model1')
    assert result == {'data': 'test'}
    
    # Statistics
    stats = cache.stats()
    assert stats['total_models'] == 1
    assert stats['memory_mb'] == 100.0
    assert stats['cache_hits'] >= 1
    assert stats['cache_misses'] >= 1
    
    cache.clear()  # Clean up


def test_cache_lru_eviction():
    """Test LRU eviction when cache is full."""
    cache = ModelCache()
    cache.clear()
    cache.max_models = 3  # Small limit for testing
    
    # Add 3 models
    cache.set('model1', 'data1', 100.0)
    cache.set('model2', 'data2', 100.0)
    cache.set('model3', 'data3', 100.0)
    
    assert cache.stats()['total_models'] == 3
    
    # Access model1 to make it recently used
    cache.get('model1')
    
    # Add 4th model - should evict model2 (least recently used)
    cache.set('model4', 'data4', 100.0)
    
    assert cache.stats()['total_models'] == 3
    assert cache.get('model1') is not None  # Still there (recently used)
    assert cache.get('model2') is None       # Evicted
    assert cache.get('model3') is not None  # Still there
    assert cache.get('model4') is not None  # Newly added
    
    cache.clear()  # Clean up


def test_cache_memory_limit():
    """Test memory limit enforcement."""
    cache = ModelCache()
    cache.clear()
    original_limit = cache.max_memory_mb
    cache.max_memory_mb = 500  # 500MB limit
    cache.max_models = 10
    
    # Add models totaling 300MB
    cache.set('model1', 'data1', 100.0)
    cache.set('model2', 'data2', 100.0)
    cache.set('model3', 'data3', 100.0)
    
    assert cache.stats()['memory_mb'] == 300.0
    
    # Try to add 300MB more (total would be 600MB > 500MB limit)
    cache.set('model4', 'data4', 300.0)
    
    # Should have evicted models to stay under limit
    stats = cache.stats()
    assert stats['memory_mb'] <= 500.0
    assert stats['evictions'] >= 1
    
    # Restore and clean up
    cache.max_memory_mb = original_limit
    cache.clear()


def test_cache_clear():
    """Test cache clearing."""
    cache = ModelCache()
    cache.clear()  # Start fresh
    
    cache.set('model1', 'data1', 100.0)
    cache.set('model2', 'data2', 100.0)
    
    assert cache.stats()['total_models'] == 2
    
    cache.clear()
    
    stats = cache.stats()
    assert stats['total_models'] == 0
    assert stats['memory_mb'] == 0.0
    assert cache.get('model1') is None
    assert cache.get('model2') is None


def test_cache_statistics():
    """Test cache statistics tracking."""
    cache = ModelCache()
    cache.clear()
    cache.reset_stats()
    
    # Generate some cache activity
    cache.get('missing')  # miss
    cache.set('model1', 'data', 100.0)
    cache.get('model1')  # hit
    cache.get('model1')  # hit
    cache.get('missing2')  # miss
    
    stats = cache.stats()
    
    assert stats['cache_hits'] >= 2
    assert stats['cache_misses'] >= 2
    assert stats['hit_rate'] > 0
    assert 'models' in stats
    assert 'model1' in stats['models']
    
    cache.clear()  # Clean up


def test_cache_disabled():
    """Test cache when disabled."""
    cache = ModelCache()
    cache.clear()
    original_setting = cache.enable_cache
    cache.enable_cache = False
    
    # Set should not cache
    cache.set('model1', 'data1', 100.0)
    
    # Get should return None
    assert cache.get('model1') is None
    
    # Re-enable
    cache.enable_cache = True
    cache.set('model2', 'data2', 100.0)
    assert cache.get('model2') == 'data2'
    
    # Restore and clean up
    cache.enable_cache = original_setting
    cache.clear()
