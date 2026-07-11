"""
Cache configuration for VeriFile-X model caching.

BUG FIX: this class previously also declared MAX_MODELS, MAX_MEMORY_MB,
ENABLE_CACHE, EVICTION_POLICY, and a to_dict() built from them — but
backend/core/model_cache.py's ModelCache hardcodes its own
max_models/max_memory_mb independently and never reads any of these,
making them a second, disconnected "source of truth" for the same
concept. Only MODEL_SIZES was ever actually read anywhere (by
dire_detector.py/clip_detector.py). Trimmed to just that.
"""


class CacheConfig:
    """Model size estimates used by dire_detector.py/clip_detector.py."""

    MODEL_SIZES = {
        'stable-diffusion-2-1': 4000,
        'clip-vit-b-32': 350,
    }
