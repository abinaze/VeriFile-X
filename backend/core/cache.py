"""
In-memory cache for forensic analysis results.

Why caching?
- Image forensics is CPU-intensive (FFT, hashing, EXIF)
- Same file uploaded twice = wasted computation
- SHA-256 hash = unique file fingerprint

Privacy note:
- Cache stores results only, never file bytes
- Results contain no personal data
- Cache cleared on server restart (no persistence)
"""
import hashlib
import threading
from typing import Dict, Optional, Any, Union
from datetime import datetime, timedelta
from backend.core.logger import setup_logger

logger = setup_logger(__name__)

# Max cached results (prevents memory abuse)
MAX_CACHE_SIZE = 500

# Cache TTL: results expire after 1 hour
CACHE_TTL_MINUTES = 60


class ForensicsCache:
    """
    Thread-safe in-memory cache for forensic results.
    Key: SHA-256 hash of file bytes
    Value: forensic report + timestamp
    
    OPTIMIZATION: Accepts pre-computed hash to avoid duplicate hashing.
    """

    def __init__(self):
        self._lock  = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        logger.info("Forensics cache initialized")

    def _compute_key(self, file_identifier: Union[bytes, str]) -> str:
        """
        Compute SHA-256 hash as cache key.
        Same file = same hash = cache hit.
        
        OPTIMIZATION: If a string (pre-computed hash) is provided,
        use it directly to avoid redundant hashing.
        
        Args:
            file_identifier: Either raw file bytes OR pre-computed SHA-256 hash
            
        Returns:
            SHA-256 hash string
        """
        if isinstance(file_identifier, str):
            # Already a hash - use directly (OPTIMIZATION)
            return file_identifier
        else:
            # Compute hash from bytes
            return hashlib.sha256(file_identifier).hexdigest()

    def get(self, file_identifier: Union[bytes, str]) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result if available and not expired.
        
        Args:
            file_identifier: Either raw file bytes OR pre-computed SHA-256 hash
            
        Returns:
            Cached report dict or None if miss/expired
        """
        key = self._compute_key(file_identifier)

        with self._lock:
            if key not in self._cache:
                logger.info(f"Cache MISS: {key[:16]}...")
                return None

            entry = self._cache[key]

            # TTL check inside lock to prevent TOCTOU race condition
            age = datetime.now() - entry.get("cached_at", datetime.now())
            if age > timedelta(minutes=CACHE_TTL_MINUTES):
                self._cache.pop(key, None)
                logger.info(f"Cache EXPIRED: {key[:16]}...")
                return None

        # Check TTL expiry
        age = datetime.now() - entry["cached_at"]
        if age > timedelta(minutes=CACHE_TTL_MINUTES):
            del self._cache[key]
            logger.info(f"Cache EXPIRED: {key[:16]}...")
            return None

        logger.info(
            f"Cache HIT: {key[:16]}... "
            f"(age={age.seconds}s, "
            f"cache_size={len(self._cache)})"
        )

        # Add cache metadata to response
        result = entry["report"].copy()
        result["cache_info"] = {
            "cached": True,
            "age_seconds": age.seconds,
            "cache_hit": True
        }
        return result

    def set(self, file_identifier: Union[bytes, str], report: Dict[str, Any]) -> None:
        """
        Store forensic report in cache.
        Evicts oldest entry if cache is full.
        
        Args:
            file_identifier: Either raw file bytes OR pre-computed SHA-256 hash
            report: Forensic analysis report to cache
        """
        # Evict oldest if at capacity
        if len(self._cache) >= MAX_CACHE_SIZE:
            oldest_key = min(
                self._cache,
                key=lambda k: self._cache[k]["cached_at"]
            )
            del self._cache[oldest_key]
            logger.info(f"Cache EVICT: {oldest_key[:16]}...")

        key = self._compute_key(file_identifier)
        self._cache[key] = {
            "report": report,
            "cached_at": datetime.now()
        }

        logger.info(
            f"Cache SET: {key[:16]}... "
            f"(cache_size={len(self._cache)})"
        )

    def size(self) -> int:
        """Return current number of cached entries."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear all cached entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache CLEARED: {count} entries removed")


# Singleton instance - shared across all requests
forensics_cache = ForensicsCache()
