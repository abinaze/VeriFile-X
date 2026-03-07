"""
Cache configuration for VeriFile-X model caching.
"""
from typing import Dict, Any


class CacheConfig:
    """Model cache configuration."""
    
    # Maximum number of models to cache
    MAX_MODELS = 10
    
    # Maximum memory usage in MB
    MAX_MEMORY_MB = 8000  # 8GB
    
    # Enable/disable caching
    ENABLE_CACHE = True
    
    # Cache eviction policy
    EVICTION_POLICY = "lru"  # least-recently-used
    
    # Model size estimates (in MB)
    MODEL_SIZES = {
        'stable-diffusion-2-1': 4000,
        'clip-vit-b-32': 350,
    }
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'max_models': cls.MAX_MODELS,
            'max_memory_mb': cls.MAX_MEMORY_MB,
            'enable_cache': cls.ENABLE_CACHE,
            'eviction_policy': cls.EVICTION_POLICY,
        }
