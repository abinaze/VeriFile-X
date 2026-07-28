"""
Cache configuration for VeriFile-X model caching.

Only MODEL_SIZES is kept -- it's the only field ModelCache actually
reads (dire_detector.py/clip_detector.py).
"""


class CacheConfig:
    """Model size estimates used by dire_detector.py/clip_detector.py."""

    MODEL_SIZES = {
        'stable-diffusion-2-1': 4000,
        'clip-vit-b-32': 350,
    }
