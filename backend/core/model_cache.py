"""
Singleton model cache manager for VeriFile-X.

Provides thread-safe model caching with LRU eviction to improve
performance by avoiding repeated model loading.
"""
import threading
from typing import Any, Dict, Optional, OrderedDict
from collections import OrderedDict as OrderedDictType
import time
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


class ModelCache:
    """Singleton model cache with LRU eviction and memory limits."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize cache on first creation."""
        if self._initialized:
            return
        
        self._cache: OrderedDictType[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_memory_mb': 0.0
        }
        
        # Configuration (can be overridden)
        self.max_models = 10
        self.max_memory_mb = 8000  # 8GB
        self.enable_cache = True
        
        self._initialized = True
        logger.info("ModelCache initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get model from cache.
        
        Args:
            key: Model identifier (e.g., 'stable-diffusion-2-1')
            
        Returns:
            Cached model or None if not found
        """
        if not self.enable_cache:
            return None
        
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._stats['hits'] += 1
                
                model_info = self._cache[key]
                logger.debug(
                    f"Cache HIT: {key} "
                    f"(size: {model_info['size_mb']:.1f}MB, "
                    f"age: {time.time() - model_info['timestamp']:.1f}s)"
                )
                return model_info['model']
            else:
                self._stats['misses'] += 1
                logger.debug(f"Cache MISS: {key}")
                return None
    
    def set(self, key: str, model: Any, size_mb: float):
        """
        Store model in cache.
        
        Args:
            key: Model identifier
            model: Model object to cache
            size_mb: Estimated memory size in MB
        """
        if not self.enable_cache:
            return
        
        with self._lock:
            # Check if we need to evict
            while (len(self._cache) >= self.max_models or 
                   self._stats['total_memory_mb'] + size_mb > self.max_memory_mb):
                if not self._cache:
                    logger.warning("Cannot cache: size exceeds max_memory_mb")
                    return
                self._evict_lru()
            
            # Store model
            self._cache[key] = {
                'model': model,
                'size_mb': size_mb,
                'timestamp': time.time()
            }
            self._cache.move_to_end(key)
            self._stats['total_memory_mb'] += size_mb
            
            logger.info(
                f"Cached model: {key} "
                f"(size: {size_mb:.1f}MB, "
                f"total: {self._stats['total_memory_mb']:.1f}MB)"
            )
    
    def _evict_lru(self):
        """Evict least recently used model."""
        if not self._cache:
            return
        
        # Remove first item (least recently used)
        key, model_info = self._cache.popitem(last=False)
        self._stats['total_memory_mb'] -= model_info['size_mb']
        self._stats['evictions'] += 1
        
        logger.info(
            f"Evicted LRU model: {key} "
            f"(freed: {model_info['size_mb']:.1f}MB)"
        )
        
        # Clean up model if it has cleanup method
        if hasattr(model_info['model'], 'cleanup'):
            try:
                model_info['model'].cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up {key}: {e}")
    
    def clear(self):
        """Clear all cached models."""
        with self._lock:
            count = len(self._cache)
            memory_freed = self._stats['total_memory_mb']
            
            # Clean up all models
            for key, model_info in self._cache.items():
                if hasattr(model_info['model'], 'cleanup'):
                    try:
                        model_info['model'].cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up {key}: {e}")
            
            self._cache.clear()
            self._stats['total_memory_mb'] = 0.0
            
            logger.info(
                f"Cleared cache: {count} models, "
                f"{memory_freed:.1f}MB freed"
            )
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache metrics
        """
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests 
                       if total_requests > 0 else 0.0)
            
            return {
                'total_models': len(self._cache),
                'memory_mb': self._stats['total_memory_mb'],
                'max_memory_mb': self.max_memory_mb,
                'cache_hits': self._stats['hits'],
                'cache_misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self._stats['evictions'],
                'enabled': self.enable_cache,
                'models': list(self._cache.keys())
            }
    
    def reset_stats(self):
        """Reset statistics counters."""
        with self._lock:
            self._stats['hits'] = 0
            self._stats['misses'] = 0
            self._stats['evictions'] = 0
            logger.info("Cache statistics reset")


# Global cache instance
_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    return _cache
