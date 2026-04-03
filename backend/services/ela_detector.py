"""
ELA (Error Level Analysis) Detection.

ELA detects inconsistencies in JPEG compression across image regions.
When an image is authentic, all regions have similar compression error levels.
When an image is AI-generated or manipulated, regions show inconsistent
error levels because they have different compression histories.

Widely used in digital forensics, journalism verification, and court cases.
"""
import numpy as np
from typing import Dict, Any
from PIL import Image, ImageChops
from io import BytesIO
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


def detect_ela(image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """
    Perform Error Level Analysis on image.

    Process:
    1. Re-save image at known JPEG quality (95)
    2. Compute pixel difference between original and re-saved
    3. Analyze the distribution of error levels across regions
    4. Inconsistent errors = manipulation or AI generation indicators
    """
    try:
        # Open original image
        original = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = original.size

        # Skip tiny images
        if width < 32 or height < 32:
            return {
                "signal_name": "ELA Compression Analysis",
                "score": 0.5,
                "confidence": 0.0,
                "explanation": "Image too small for ELA analysis",
                "method": "ela"
            }

        # Re-save at known quality
        buffer = BytesIO()
        original.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        # Compute difference
        diff = ImageChops.difference(original, recompressed)
        diff_array = np.array(diff, dtype=np.float64)

        # === Signal 1: Global error level ===
        # AI images: very uniform low error (synthesized at consistent quality)
        # Real photos: moderate variation in error levels
        mean_error = float(np.mean(diff_array))

        # === Signal 2: Regional inconsistency ===
        # Divide image into blocks and measure error variance between blocks
        block_size = max(16, min(width, height) // 8)
        block_means = []

        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = diff_array[y:y+block_size, x:x+block_size]
                block_means.append(float(np.mean(block)))

        if len(block_means) > 4:
            block_mean = float(np.mean(block_means))
            # Coefficient of variation: how inconsistent are regions?
            cv = float(np.std(block_means) / (block_mean + 1e-10))
        else:
            block_variance = 0.0
            cv = 0.0

        # === Signal 3: High error region concentration ===
        # AI images: error concentrated in specific patterns (e.g. edges)
        # Real photos: error distributed across image
        flat = diff_array.flatten()
        high_error_pct = float(np.sum(flat > np.percentile(flat, 90)) / len(flat))
        error_concentration = abs(high_error_pct - 0.10)  # Expected ~10% above 90th pct

        # === Combine into AI score ===
        # Very low mean error + low variance = likely AI (uniform synthesis)
        # Very high variance = likely manipulation

        # Normalize mean error (real photos: typically 3-15, AI: 0.5-5)
        if mean_error < 1.5:
            mean_score = 0.8  # Very low error = AI signature
        elif mean_error < 4.0:
            mean_score = 0.5
        elif mean_error < 10.0:
            mean_score = 0.3  # Normal photo range
        else:
            mean_score = 0.4  # High error = possibly edited

        # High coefficient of variation = inconsistent regions = manipulation
        if cv > 2.0:
            cv_score = 0.75  # Very inconsistent = manipulation
        elif cv > 1.0:
            cv_score = 0.55
        else:
            cv_score = 0.25  # Consistent = real or clean AI

        # Error concentration anomaly
        concentration_score = min(1.0, error_concentration * 5)

        # Weighted combination
        ai_score = (
            0.50 * mean_score +
            0.30 * cv_score +
            0.20 * concentration_score
        )
        ai_score = float(np.clip(ai_score, 0.0, 1.0))

        # Confidence based on image size
        pixel_count = width * height
        confidence = min(0.80, 0.4 + (pixel_count / (512 * 512)) * 0.40)

        if mean_error < 2.0:
            explanation = (
                f"Very low ELA error ({mean_error:.2f}) — "
                "uniform compression consistent with AI synthesis"
            )
        elif cv > 1.5:
            explanation = (
                f"High regional ELA inconsistency (CV={cv:.2f}) — "
                "compression anomalies detected across image regions"
            )
        else:
            explanation = (
                f"Normal ELA pattern (mean={mean_error:.2f}, CV={cv:.2f}) — "
                "compression levels consistent with authentic photo"
            )

        logger.info(
            f"ELA detection: score={ai_score:.3f}, "
            f"mean_err={mean_error:.2f}, cv={cv:.2f}, file={filename}"
        )

        return {
            "signal_name": "ELA Compression Analysis",
            "score": ai_score,
            "confidence": confidence,
            "explanation": explanation,
            "raw_value": mean_error,
            "expected_range": "< 2.0 mean error for AI images",
            "method": "ela_jpeg"
        }

    except Exception as e:
        logger.warning(f"ELA detection failed: {e}")
        return {
            "signal_name": "ELA Compression Analysis",
            "score": 0.5,
            "confidence": 0.0,
            "explanation": f"ELA analysis unavailable: {str(e)}",
            "raw_value": 0.0,
            "method": "ela_jpeg"
        }
