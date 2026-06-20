"""
Noise Map Detection — Signal 28.

Extracts and analyses the high-frequency noise residual of an image
to distinguish authentic camera noise from synthetic texture patterns.

Theory
------
Digital camera sensors introduce two classes of noise:
  1. Fixed-pattern noise (FPN) — deterministic per-sensor imperfections
  2. Temporal / photon shot noise — random per-exposure, Poisson-distributed

Both produce characteristic statistical fingerprints in the noise residual
(original − smoothed).  Key properties of real camera noise:
  - Spatial variance correlates with local luminance (shot noise)
  - Residual frequency spectrum follows ~1/f decay (pink noise)
  - Adjacent-pixel correlation is low and spatially uniform
  - Histogram is approximately Gaussian with zero mean

AI-generated images violate these properties in measurable ways:
  - Noise residual is spatially uniform (no luminance correlation)
  - Spectrum is flat or has harmonic peaks (up-convolution artefacts)
  - Some regions show zero variance (pure synthetic texture)
  - Histogram may be multimodal or have heavy tails (GAN artefacts)

Algorithm
---------
1. Convert to float32 grayscale.
2. Smooth with a Gaussian kernel (σ=1.0) — the "scene content" estimate.
3. Noise map = original − smoothed.
4. Compute five sub-signals from the residual:
   a. Spatial variance map: 16×16 patch variance; measure correlation
      with patch luminance (Pearson r² — high = real, low = AI)
   b. Frequency spectrum: 2D FFT radial power profile; fit 1/f slope
      (steep = real camera, flat = AI synthesiser)
   c. Zero-variance patch fraction: patches with σ < 0.5 (AI over-smooth)
   d. Residual histogram kurtosis: excess kurtosis (Gaussian ≈ 0)
   e. Inter-patch autocorrelation regularity (AI up-convolution artefact)
5. Combine sub-signals into a final [0,1] AI score.

References
----------
- Mahdian & Saic (2009): "Using noise inconsistencies for blind image forensics"
- Cozzolino et al. (2019): "Noiseprint: A CNN-Based Camera Model Fingerprint"
- Liu et al. (2020): "Global Texture Enhancement for Fake Face Detection"
"""
import numpy as np
from io import BytesIO
from typing import Dict, Any, Tuple

from PIL import Image
from backend.core.logger import setup_logger

logger = setup_logger(__name__)

# Gaussian smoothing kernel σ for noise extraction
_GAUSS_SIGMA = 1.0

# Patch size for spatial analysis
_PATCH = 16

# Minimum patches needed for meaningful spatial statistics
_MIN_PATCHES = 9

# Zero-variance threshold (if std < this the patch is considered flat)
_ZERO_VAR_THRESH = 0.5


# ── Utilities ──────────────────────────────────────────────────────────────────

def _gaussian_smooth(arr: np.ndarray, sigma: float = _GAUSS_SIGMA) -> np.ndarray:
    """Apply Gaussian smoothing; falls back to uniform filter if scipy absent."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr, sigma=sigma).astype(np.float32)
    except ImportError:
        from scipy.ndimage import uniform_filter
        ksize = max(3, int(sigma * 3) | 1)  # ensure odd
        return uniform_filter(arr, size=ksize).astype(np.float32)


def _extract_noise_map(gray: np.ndarray) -> np.ndarray:
    """Return noise residual = original − Gaussian-smoothed."""
    smoothed = _gaussian_smooth(gray, _GAUSS_SIGMA)
    return (gray - smoothed).astype(np.float32)


def _patch_stats(
    gray: np.ndarray, noise: np.ndarray, patch: int = _PATCH
) -> Tuple[float, float, float]:
    """
    Per-patch analysis.

    Returns:
      lum_noise_corr   — Pearson r between patch luminance and patch noise-std
                         (high ≈ shot noise present ≈ real)
      zero_var_frac    — fraction of patches with near-zero noise variance
      mean_patch_var   — average noise variance across patches
    """
    h, w = gray.shape
    lum_vals, noise_std_vals = [], []
    zero_count, total = 0, 0

    for y in range(0, h - patch, patch):
        for x in range(0, w - patch, patch):
            g_patch = gray[y:y + patch, x:x + patch]
            n_patch = noise[y:y + patch, x:x + patch]
            lum = float(np.mean(g_patch))
            nstd = float(np.std(n_patch))
            lum_vals.append(lum)
            noise_std_vals.append(nstd)
            if nstd < _ZERO_VAR_THRESH:
                zero_count += 1
            total += 1

    if total < _MIN_PATCHES:
        return 0.5, 0.0, 1.0

    lum_arr = np.array(lum_vals, dtype=np.float64)
    std_arr = np.array(noise_std_vals, dtype=np.float64)
    mean_var = float(np.mean(std_arr ** 2))

    # Pearson correlation coefficient
    if lum_arr.std() < 1e-6 or std_arr.std() < 1e-6:
        corr = 0.0
    else:
        corr = float(np.corrcoef(lum_arr, std_arr)[0, 1])

    zero_frac = zero_count / total
    return float(np.clip(corr, -1.0, 1.0)), zero_frac, mean_var


def _frequency_slope(noise: np.ndarray) -> float:
    """
    Fit 1/f slope to the radial power spectrum of the noise residual.

    Real camera noise: slope ≈ −1.0 to −2.0 (pink / red noise).
    AI synthesisers: slope closer to 0 (flat / white noise) or
    positive (excess high-frequency texture from up-convolution).

    Returns the fitted slope (raw, not clipped).
    """
    h, w = noise.shape
    fft = np.fft.fft2(noise)
    power = np.abs(np.fft.fftshift(fft)) ** 2

    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)

    max_r = min(cx, cy)
    radii, mean_powers = [], []
    for r in range(1, max_r, max(1, max_r // 40)):
        mask = (R >= r) & (R < r + max(1, max_r // 40))
        if mask.sum() == 0:
            continue
        radii.append(np.log(float(r) + 1e-8))
        mean_powers.append(np.log(float(power[mask].mean()) + 1e-8))

    if len(radii) < 6:
        return 0.0

    slope = float(np.polyfit(radii, mean_powers, 1)[0])
    return slope


def _residual_kurtosis(noise: np.ndarray) -> float:
    """
    Excess kurtosis of the noise residual histogram.
    Gaussian (real noise) ≈ 0; GAN artefacts often show |kurtosis| > 2.
    """
    flat = noise.flatten().astype(np.float64)
    if flat.std() < 1e-6:
        return 0.0
    n = len(flat)
    mean = flat.mean()
    std = flat.std()
    kurt = float(np.mean(((flat - mean) / std) ** 4)) - 3.0
    return kurt


def _inter_patch_regularity(noise: np.ndarray, patch: int = _PATCH) -> float:
    """
    Measure spatial regularity of the noise variance map.

    AI up-convolution (e.g. transpose-conv) creates a grid pattern with
    period equal to the stride. This appears as a regular tiling in the
    variance map — measurable as the ratio of dominant periodogram peak
    to median periodogram value.

    Returns regularity score [0, 1]: high = regular (AI-like).
    """
    h, w = noise.shape
    var_map_rows, var_map_cols = [], []

    for y in range(0, h - patch, patch):
        row_vars = []
        for x in range(0, w - patch, patch):
            n_patch = noise[y:y + patch, x:x + patch]
            row_vars.append(float(np.var(n_patch)))
        if row_vars:
            var_map_rows.append(row_vars)

    if len(var_map_rows) < 3 or len(var_map_rows[0]) < 3:
        return 0.0

    var_grid = np.array(var_map_rows, dtype=np.float64)
    fft2 = np.abs(np.fft.fft2(var_grid))
    fft2[0, 0] = 0.0  # remove DC

    if fft2.max() < 1e-10:
        return 0.0

    peak = float(fft2.max())
    median = float(np.median(fft2[fft2 > 0]))
    regularity = float(np.clip((peak / (median + 1e-10) - 1.0) / 20.0, 0.0, 1.0))
    return regularity


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_noise_map(image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """
    Run Noise Map analysis on *image_bytes*.

    Five sub-signals are computed and combined into an AI probability:
      1. Luminance-noise correlation  (high → real, low → AI)
      2. Frequency spectrum slope     (steep negative → real, flat → AI)
      3. Zero-variance patch fraction (high → AI over-smoothing)
      4. Residual kurtosis            (|high| → AI artefact)
      5. Inter-patch regularity       (high → AI up-convolution grid)

    Returns signal dict compatible with the ensemble detector.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("L")
        gray = np.array(img, dtype=np.float32)
        h, w = gray.shape

        if h < 64 or w < 64:
            return {
                "signal_name": "Noise Map Analysis",
                "score": 0.5,
                "confidence": 0.0,
                "explanation": "Image too small for noise map analysis",
                "method": "noise_map",
            }

        noise = _extract_noise_map(gray)

        # Sub-signal 1: luminance–noise correlation
        lum_corr, zero_var_frac, mean_patch_var = _patch_stats(gray, noise)
        # High positive correlation = shot noise present = real camera
        # Score: high corr → low AI score
        corr_score = float(np.clip(0.5 - lum_corr * 0.5, 0.0, 1.0))

        # Sub-signal 2: frequency spectrum slope
        slope = _frequency_slope(noise)
        # Real camera: slope ~ −1.5 to −2.5. AI: slope ~ −0.5 to +0.5
        # Map: slope < −1.0 → low AI score; slope > −0.5 → high AI score
        slope_score = float(np.clip((slope + 1.5) / 2.0, 0.0, 1.0))

        # Sub-signal 3: zero-variance patch fraction
        # High fraction → AI over-smoothing
        zero_score = float(np.clip(zero_var_frac * 2.0, 0.0, 1.0))

        # Sub-signal 4: kurtosis deviation
        kurtosis = _residual_kurtosis(noise)
        # Real camera noise: kurtosis ≈ 0–1. GAN: often |kurtosis| > 2
        kurt_score = float(np.clip(abs(kurtosis) / 5.0, 0.0, 1.0))

        # Sub-signal 5: inter-patch regularity (up-convolution grid)
        regularity = _inter_patch_regularity(noise)
        reg_score = regularity  # already [0,1]

        # Weighted combination
        ai_score = (
            0.30 * corr_score +
            0.25 * slope_score +
            0.20 * zero_score +
            0.15 * kurt_score +
            0.10 * reg_score
        )
        ai_score = float(np.clip(ai_score, 0.0, 1.0))

        # Confidence: driven by how many sub-signals agree
        sub_scores = [corr_score, slope_score, zero_score, kurt_score, reg_score]
        agreement = 1.0 - float(np.std(sub_scores))
        confidence = float(np.clip(agreement * 0.85, 0.05, 0.85))

        # Human-readable explanation
        if ai_score > 0.70:
            explanation = (
                f"Noise map shows AI synthesis patterns: "
                f"luminance-noise correlation r={lum_corr:.2f} (low), "
                f"spectrum slope={slope:.2f} (flat), "
                f"zero-variance patches={zero_var_frac:.0%}. "
                "These are inconsistent with authentic camera sensor noise."
            )
        elif ai_score < 0.35:
            explanation = (
                f"Noise map consistent with authentic camera sensor: "
                f"luminance-noise correlation r={lum_corr:.2f} (healthy), "
                f"spectrum slope={slope:.2f} (pink noise profile), "
                f"inter-patch regularity={regularity:.2f}."
            )
        else:
            explanation = (
                f"Noise map inconclusive: slope={slope:.2f}, "
                f"corr={lum_corr:.2f}, zero-var={zero_var_frac:.0%}, "
                f"kurtosis={kurtosis:.2f}. "
                "Could be heavily processed authentic image or subtle AI synthesis."
            )

        return {
            "signal_name":    "Noise Map Analysis",
            "score":          round(ai_score, 4),
            "confidence":     round(confidence, 4),
            "explanation":    explanation,
            "method":         "noise_map",
            "raw_value":      round(ai_score, 4),
            "expected_range": "0.0–0.4 real camera; > 0.6 AI synthetic",
            "sub_signals": {
                "luminance_noise_corr": round(lum_corr, 4),
                "freq_slope":           round(slope, 4),
                "zero_var_frac":        round(zero_var_frac, 4),
                "kurtosis":             round(kurtosis, 4),
                "regularity":           round(regularity, 4),
            },
        }

    except Exception:
        logger.error("Noise Map detection failed for %s", filename, exc_info=True)
        return {
            "signal_name": "Noise Map Analysis",
            "score": 0.5,
            "confidence": 0.0,
            "explanation": "Noise map analysis failed",
            "method": "noise_map",
        }
