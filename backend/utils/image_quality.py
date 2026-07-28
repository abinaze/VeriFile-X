"""
Image Quality Gate for VeriFile-X.

Checks image quality before running expensive forensic analysis.
Rejects or flags images that are too small, corrupt, or unreadable.

Quality tiers:
  good       - Full analysis, all signals active
  degraded   - Analysis runs but some signals unreliable
  low        - Analysis runs with reduced confidence
  unsuitable - Reject (too small, corrupt)
"""
from backend.core.logger import setup_logger
from typing import Dict, Any
from io import BytesIO
from PIL import Image

logger = setup_logger(__name__)

# Guard against decompression bombs (e.g. 1KB PNG -> 4GB bitmap), set at
# import time (F-18) rather than inside assess_image_quality() -- this
# module used to set it only when that function ran, but
# analyze_image()'s EXIF-orientation-correction step (Image.open() +
# exif_transpose() + .save(), which fully decodes and re-encodes pixel
# data -- exactly the operation this limit exists to protect) ran
# BEFORE assess_image_quality() in the request handler, under Pillow's
# default ~89-megapixel ceiling instead of this module's intended,
# tighter 50-megapixel one. Setting it here means it takes effect the
# moment this module is imported, before any request is ever handled,
# rather than only after the first call to assess_image_quality().
Image.MAX_IMAGE_PIXELS = 50_000_000

MIN_WIDTH  = 64
MIN_HEIGHT = 64

RECOMMENDED_MIN_WIDTH  = 256
RECOMMENDED_MIN_HEIGHT = 256


def assess_image_quality(image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
    from PIL import Image
    warnings = []

    try:
        img          = Image.open(BytesIO(image_bytes))
        width, height = img.size

        # Explicit upper-bound check (belt + suspenders alongside MAX_IMAGE_PIXELS)
        if width > 10_000 or height > 10_000:
            return {
                "tier": "unsuitable", "suitable": False,
                "width": width, "height": height, "pixel_count": width * height,
                "format": img.format or "unknown", "mode": img.mode,
                "warnings": [f"Image too large ({width}x{height}). Maximum: 10000x10000"],
                "confidence_cap": 0.0,
                "reason": f"Image dimensions {width}x{height} exceed the 10000×10000 analysis limit.",
            }
        fmt          = img.format or "unknown"
        mode         = img.mode
        pixels       = width * height
    except Exception:
        logger.warning("Image quality check failed for %s", filename)
        return {
            "tier": "unsuitable", "suitable": False,
            "width": 0, "height": 0, "pixel_count": 0,
            "format": "unknown", "mode": "unknown",
            "warnings": ["Image could not be decoded"],
            "confidence_cap": 0.0,
            "reason": "Image is corrupt or unreadable",
        }

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return {
            "tier": "unsuitable", "suitable": False,
            "width": width, "height": height, "pixel_count": pixels,
            "format": fmt, "mode": mode,
            "warnings": [f"Image too small ({width}x{height}). Minimum: {MIN_WIDTH}x{MIN_HEIGHT}"],
            "confidence_cap": 0.0,
            "reason": f"Image {width}x{height} is below minimum {MIN_WIDTH}x{MIN_HEIGHT} for forensic analysis.",
        }

    if width < RECOMMENDED_MIN_WIDTH or height < RECOMMENDED_MIN_HEIGHT:
        warnings.append(f"Image is small ({width}x{height}). Less reliable below {RECOMMENDED_MIN_WIDTH}x{RECOMMENDED_MIN_HEIGHT}.")

    if mode in ("1", "P"):
        warnings.append(f"Color mode {mode} has limited forensic information.")
    if mode == "L":
        warnings.append("Grayscale image — color-based signals will be inactive.")
    if mode == "RGBA":
        warnings.append("RGBA image — alpha channel ignored.")

    bytes_per_pixel = len(image_bytes) / max(pixels, 1)
    if fmt == "JPEG" and bytes_per_pixel < 0.3:
        warnings.append(f"Heavily compressed JPEG ({bytes_per_pixel:.2f} bytes/pixel) may interfere with some signals.")

    if width < RECOMMENDED_MIN_WIDTH or height < RECOMMENDED_MIN_HEIGHT:
        tier, confidence_cap = "low", 0.70
    elif warnings:
        tier, confidence_cap = "degraded", 0.85
    else:
        tier, confidence_cap = "good", 1.0

    logger.info("Image quality %s: tier=%s (%dx%d %s %s)", filename, tier, width, height, fmt, mode)

    return {
        "tier": tier, "suitable": True,
        "width": width, "height": height, "pixel_count": pixels,
        "format": fmt, "mode": mode,
        "warnings": warnings, "confidence_cap": confidence_cap, "reason": None,
    }
