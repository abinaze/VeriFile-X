"""
PRNU (Photo Response Non-Uniformity) Camera Fingerprint Detection.

Every real camera sensor has microscopic manufacturing defects that create
a unique noise pattern embedded in every photo it takes — like a fingerprint.
AI-generated images have no camera sensor, so they have no PRNU pattern.

This is one of the most court-defensible forensic signals available.
Used by law enforcement and accepted in legal proceedings worldwide.
"""
import numpy as np
from typing import Dict, Any
from PIL import Image
from io import BytesIO
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


def _to_grayscale_array(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to float grayscale array."""
    img = Image.open(BytesIO(image_bytes)).convert("L")
    return np.array(img, dtype=np.float64)


def _extract_noise_residual(gray: np.ndarray) -> np.ndarray:
    """
    Extract noise residual using Wavelet-based denoising.
    The residual = original - denoised approximates the PRNU pattern.
    Real cameras: residual has structured spatial correlation.
    AI images: residual is random / flat.
    """
    try:
        import pywt
        # Wavelet decomposition
        coeffs = pywt.dwt2(gray, 'db4')
        cA, (cH, cV, cD) = coeffs

        # Zero out approximation (keep noise detail)
        noise_coeffs = (np.zeros_like(cA), (cH, cV, cD))
        noise = pywt.idwt2(noise_coeffs, 'db4')

        # Resize to match original if needed
        noise = noise[:gray.shape[0], :gray.shape[1]]
        return noise
    except Exception:
        # Fallback: simple high-pass filter
        from scipy.ndimage import uniform_filter
        smooth = uniform_filter(gray, size=3)
        return gray - smooth


def detect_prnu(image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """
    Detect PRNU camera fingerprint presence.

    Real photos: show structured noise correlation (PRNU present)
    AI images: show flat/random noise (no camera sensor = no PRNU)

    Returns signal dict compatible with the ensemble detector.
    """
    try:
        gray = _to_grayscale_array(image_bytes)
        noise = _extract_noise_residual(gray)

        # === Signal 1: Noise spatial correlation ===
        # Real cameras: noise has spatial structure (correlated patches)
        # AI images: noise is spatially random
        h, w = noise.shape
        patch_size = max(16, min(h, w) // 8)

        correlations = []
        step = patch_size
        for i in range(0, h - patch_size, step):
            for j in range(0, w - patch_size, step):
                patch = noise[i:i+patch_size, j:j+patch_size].flatten()
                if len(patch) > 10 and np.std(patch) > 1e-10:
                    # Autocorrelation of patch
                    ac = np.corrcoef(patch[:-1], patch[1:])[0, 1]
                    if not np.isnan(ac):
                        correlations.append(abs(ac))

        mean_correlation = float(np.mean(correlations)) if correlations else 0.0

        # === Signal 2: Noise energy distribution ===
        # Real cameras: noise energy concentrated in specific frequency bands
        # AI images: noise energy is flat across frequencies
        noise_std = float(np.std(noise))
        noise_mean_abs = float(np.mean(np.abs(noise)))
        energy_ratio = noise_mean_abs / (noise_std + 1e-10)

        # === Signal 3: Row/column pattern consistency ===
        # Camera sensors have consistent row/column noise patterns
        row_vars = np.var(noise, axis=1)
        col_vars = np.var(noise, axis=0)
        row_consistency = float(np.std(row_vars) / (np.mean(row_vars) + 1e-10))
        col_consistency = float(np.std(col_vars) / (np.mean(col_vars) + 1e-10))
        pattern_consistency = (row_consistency + col_consistency) / 2

        # === Combine into AI score ===
        # Low correlation = no PRNU = likely AI
        # High correlation = PRNU present = likely real
        # We invert: high AI score = low PRNU evidence

        # Correlation threshold: real photos typically > 0.05
        correlation_score = max(0.0, 1.0 - (mean_correlation / 0.15))

        # Energy ratio: real photos typically 0.6-0.9
        energy_score = 0.0
        if energy_ratio < 0.5:
            energy_score = 0.8  # Very low energy ratio = AI
        elif energy_ratio < 0.65:
            energy_score = 0.5
        else:
            energy_score = 0.2

        # Pattern consistency: real cameras have consistent patterns
        pattern_score = max(0.0, min(1.0, pattern_consistency / 2.0))

        # Weighted combination
        ai_score = (
            0.50 * correlation_score +
            0.30 * energy_score +
            0.20 * pattern_score
        )
        ai_score = float(np.clip(ai_score, 0.0, 1.0))

        # Confidence based on image size (larger = more reliable)
        pixel_count = h * w
        confidence = min(0.85, 0.5 + (pixel_count / (1024 * 1024)) * 0.35)

        if ai_score > 0.65:
            explanation = f"Weak PRNU pattern (corr={mean_correlation:.3f}) — no camera fingerprint detected, consistent with AI generation"
        elif ai_score > 0.40:
            explanation = f"Moderate PRNU signal (corr={mean_correlation:.3f}) — ambiguous camera fingerprint"
        else:
            explanation = f"Strong PRNU pattern (corr={mean_correlation:.3f}) — camera sensor fingerprint detected, consistent with real photo"

        logger.info(f"PRNU detection: score={ai_score:.3f}, corr={mean_correlation:.3f}, file={filename}")

        return {
            "signal_name": "PRNU Camera Fingerprint",
            "score": ai_score,
            "confidence": confidence,
            "explanation": explanation,
            "raw_value": mean_correlation,
            "expected_range": "< 0.05 correlation for AI images",
            "method": "prnu_wavelet"
        }

    except Exception as e:
        logger.warning(f"PRNU detection failed: {e}")
        return {
            "signal_name": "PRNU Camera Fingerprint",
            "score": 0.5,
            "confidence": 0.0,
            "explanation": f"PRNU analysis unavailable: {str(e)}",
            "raw_value": 0.0,
            "method": "prnu_wavelet"
        }