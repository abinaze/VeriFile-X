"""
CFA (Color Filter Array) Artifact Analysis — Phase 22 (Signal 30).

Every digital camera sensor uses a Bayer CFA — a mosaic of red, green,
and blue filters. The sensor captures only one colour per pixel; the other
two are interpolated (demosaiced). This demosaicing creates characteristic
inter-pixel correlations in the green channel that are unique to real cameras.

AI-generated images are rendered from floating-point pixel data with no
Bayer sensor, so they have NO demosaicing correlations — the inter-pixel
pattern is absent or random.

Algorithm (from Popescu & Farid, 2005 + green-channel variant):
  1. Extract the green channel (most abundant in Bayer pattern — 50% of pixels).
  2. Compute skip-0 diff: adjacent pixel differences  green[:, :-1] - green[:, 1:]
  3. Compute skip-1 diff: alternate pixel differences green[:, :-2] - green[:, 2:]
  4. cfa_ratio = std(skip0) / std(skip1)
     Real cameras: skip0 >> skip1 → ratio < 1.0 (demosaicing smooths neighbours)
     AI images:    skip0 ≈ skip1 → ratio ≈ 1.0  (no Bayer pattern)
  5. Also analyse row-direction for asymmetric Bayer layouts.
  6. Score = how close the ratio is to 1.0.

Reference:
  Popescu & Farid (2005) "Exposing digital forgeries in color filter array
  interpolated images", IEEE Trans. Signal Processing.
"""
import numpy as np
from io import BytesIO
from typing import Dict, Any
from PIL import Image

from backend.core.logger import setup_logger

logger = setup_logger(__name__)

_MIN_DIMENSION = 32

# Real-camera calibration band: empirically cfa_ratio ∈ [0.55, 0.90]
_REAL_RATIO_LOW  = 0.55
_REAL_RATIO_HIGH = 0.90


def detect_cfa_artifacts(image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """
    Detect presence/absence of Bayer CFA demosaicing correlations.

    Returns a detection signal dict compatible with the VeriFile-X ensemble.
    Score → 1.0 = strong AI indicator (no CFA pattern detected).
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float64)
        h, w, _ = arr.shape

        if h < _MIN_DIMENSION or w < _MIN_DIMENSION:
            return _fallback("Image too small for CFA analysis")

        green = arr[:, :, 1]  # Green channel

        # ── Column-direction CFA ratio ────────────────────────────────────────
        skip0_col = green[:, :-1] - green[:, 1:]     # adjacent columns
        skip1_col = green[:, :-2] - green[:, 2:]     # alternate columns

        std_skip0_col = float(np.std(skip0_col))
        std_skip1_col = float(np.std(skip1_col))

        cfa_ratio_col = (
            std_skip0_col / std_skip1_col
            if std_skip1_col > 1e-10
            else 1.0
        )

        # ── Row-direction CFA ratio ───────────────────────────────────────────
        skip0_row = green[:-1, :] - green[1:, :]     # adjacent rows
        skip1_row = green[:-2, :] - green[2:, :]     # alternate rows

        std_skip0_row = float(np.std(skip0_row))
        std_skip1_row = float(np.std(skip1_row))

        cfa_ratio_row = (
            std_skip0_row / std_skip1_row
            if std_skip1_row > 1e-10
            else 1.0
        )

        # Average both directions
        cfa_ratio = (cfa_ratio_col + cfa_ratio_row) / 2.0

        # ── Sub-band spectral analysis ────────────────────────────────────────
        # Real cameras: green channel has periodic spectral peaks at Bayer freq.
        # AI images: flat spectrum.
        fft = np.fft.fft2(green)
        fft_mag = np.abs(np.fft.fftshift(fft))
        fft_mag_norm = fft_mag / (np.max(fft_mag) + 1e-10)

        # Peak-to-mean ratio of FFT magnitude
        fft_peak_ratio = float(np.max(fft_mag_norm) / (np.mean(fft_mag_norm) + 1e-10))
        # High peak → periodic pattern → real camera. Low peak → flat → AI.
        # Normalise: real ≈ 10–50, AI ≈ 2–8
        spectral_score = float(np.clip(1.0 - (fft_peak_ratio - 2.0) / 30.0, 0.0, 1.0))

        # ── CFA ratio → AI score ─────────────────────────────────────────────
        # Real camera: cfa_ratio ∈ [0.55, 0.90]
        # AI / no-CFA: cfa_ratio close to 1.0 (or > 1.0)
        if cfa_ratio <= _REAL_RATIO_LOW:
            # Very low ratio — strong demosaicing present → authentic
            ratio_score = 0.10
        elif cfa_ratio <= _REAL_RATIO_HIGH:
            # Within real-camera band — linear interpolation
            ratio_score = (cfa_ratio - _REAL_RATIO_LOW) / (_REAL_RATIO_HIGH - _REAL_RATIO_LOW) * 0.50
        else:
            # Above real band → no CFA → AI indicator
            excess = min(cfa_ratio - _REAL_RATIO_HIGH, 0.30)
            ratio_score = 0.50 + (excess / 0.30) * 0.50

        ratio_score = float(np.clip(ratio_score, 0.0, 1.0))

        # ── Combined score ────────────────────────────────────────────────────
        ai_score = float(np.clip(
            0.70 * ratio_score + 0.30 * spectral_score,
            0.0, 1.0,
        ))

        # Confidence: larger images give more reliable CFA statistics
        pixel_count = h * w
        confidence = float(np.clip(0.35 + (pixel_count / 2_000_000) * 0.50, 0.35, 0.85))

        # ── Explanation ───────────────────────────────────────────────────────
        if cfa_ratio < _REAL_RATIO_LOW:
            explanation = (
                f"Strong Bayer CFA demosaicing pattern detected "
                f"(cfa_ratio={cfa_ratio:.3f}) — "
                "consistent with real camera sensor interpolation"
            )
        elif cfa_ratio <= _REAL_RATIO_HIGH:
            explanation = (
                f"Moderate CFA correlations (cfa_ratio={cfa_ratio:.3f}) — "
                "within typical real-camera range; some processing may have reduced pattern"
            )
        else:
            explanation = (
                f"Absent or weak CFA pattern (cfa_ratio={cfa_ratio:.3f}) — "
                "no Bayer demosaicing correlations detected, "
                "consistent with AI-generated or heavily processed image"
            )

        logger.info(
            "CFA: file=%s ratio_col=%.3f ratio_row=%.3f ratio=%.3f score=%.3f",
            filename, cfa_ratio_col, cfa_ratio_row, cfa_ratio, ai_score,
        )

        return {
            "signal_name":    "CFA Artifact Analysis",
            "score":          ai_score,
            "confidence":     confidence,
            "explanation":    explanation,
            "raw_value":      cfa_ratio,
            "expected_range": "cfa_ratio < 0.90 for real camera images",
            "method":         "cfa_bayer",
            "cfa_ratio":      round(cfa_ratio, 4),
            "cfa_ratio_col":  round(cfa_ratio_col, 4),
            "cfa_ratio_row":  round(cfa_ratio_row, 4),
        }

    except Exception as exc:
        logger.warning("CFA analysis failed for %s: %s", filename, exc, exc_info=True)
        return _fallback(f"CFA analysis unavailable: {exc}")


def _fallback(reason: str) -> Dict[str, Any]:
    return {
        "signal_name": "CFA Artifact Analysis",
        "score":       0.5,
        "confidence":  0.0,
        "explanation": reason,
        "raw_value":   0.0,
        "method":      "cfa_bayer",
    }
