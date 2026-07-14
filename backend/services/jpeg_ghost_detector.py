"""
JPEG Ghost Detection — Signal 27.

JPEG Ghost is a forensic technique that reveals the original JPEG
compression quality of an image by re-compressing it at every quality
level from 51–99 and measuring normalised squared error (NSE).

Principle
---------
When an image was previously saved at quality Q, re-compressing it at
Q produces almost zero additional error — it has already reached the
minimum energy state for that quantisation table.  Re-compressing at
any other quality produces markedly higher error.  The quality level
that produces the *minimum* NSE is therefore the original compression
quality — the "ghost".

Forensic implication
--------------------
AI-generated images are typically rendered from floating-point pixel
data with no prior JPEG history.  When re-compressed, they show either:
  (a) no ghost at all — flat NSE curve (synthetic images)
  (b) an extremely strong ghost at a single quality level suspicious
      of being added deliberately post-generation.

Real photographs show a recognisable ghost trough at their original
capture quality.  Images that have been manipulated often show a
*ghost mismatch* — the ghost quality disagrees with the stored EXIF
quality or shows a bimodal trough (two compression histories).

References
----------
- Farid H. (2009): "A Picture Tells a Thousand Lies"
- Kee & Farid (2011): "Detecting Doctored Images Using Camera Response Normality and Consistency"
- Christlein et al. (2012): "An Evaluation of Popular Copy-Move Forgery Detection Approaches"
"""
import numpy as np
from io import BytesIO
from typing import Dict, Any, List, Tuple

from PIL import Image
from backend.core.logger import setup_logger

logger = setup_logger(__name__)

# Quality levels to probe (51–99 inclusive, step 2 for speed; full scan = step 1)
# Extended to Q=30 to catch WhatsApp (~50) and Instagram (~60-75) recompression
_Q_RANGE = range(30, 100, 2)

# A flat NSE curve (std/mean < threshold) means no prior JPEG history
_FLAT_CURVE_THRESHOLD = 0.06

# Ghost trough depth: ratio of minimum to median NSE — strong ghost if < this
_STRONG_GHOST_RATIO = 0.45

# Suspicious ghost strength if the minimum is exceptionally isolated
_ISOLATED_TROUGH_ZSCORE = 2.5


def _image_nse_at_quality(
    original: np.ndarray, pil_img: Image.Image, quality: int
) -> float:
    """
    Re-save *pil_img* at *quality*, reload, and return normalised squared error
    against the float32 original pixel array.
    """
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
    diff = original - recomp
    nse = float(np.mean(diff ** 2))
    return nse


def _scan_nse_curve(
    pil_img: Image.Image,
) -> Tuple[List[int], List[float]]:
    """Return (quality_levels, nse_values) for all probed qualities."""
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
    qualities: List[int] = []
    nse_values: List[float] = []
    for q in _Q_RANGE:
        nse = _image_nse_at_quality(arr, pil_img, q)
        qualities.append(q)
        nse_values.append(nse)
    return qualities, nse_values


def _analyse_nse_curve(
    qualities: List[int], nse_values: List[float]
) -> Dict[str, Any]:
    """
    Interpret the NSE curve.

    Returns a dict with:
      ghost_quality        — quality level of the NSE minimum
      ghost_ratio          — min_nse / median_nse  (lower = stronger ghost)
      curve_flatness       — std / (mean + ε)  (higher = more uniform = no ghost)
      trough_z_score       — how many σ below the mean the minimum sits
      verdict              — 'strong_ghost' | 'weak_ghost' | 'no_ghost'
    """
    arr = np.array(nse_values, dtype=np.float64)
    min_idx = int(np.argmin(arr))
    min_nse = float(arr[min_idx])
    median_nse = float(np.median(arr))
    mean_nse = float(np.mean(arr))
    std_nse = float(np.std(arr))

    ghost_quality = qualities[min_idx]
    ghost_ratio = min_nse / (median_nse + 1e-10)
    curve_flatness = std_nse / (mean_nse + 1e-10)
    trough_z = (mean_nse - min_nse) / (std_nse + 1e-10)

    if curve_flatness < _FLAT_CURVE_THRESHOLD:
        verdict = "no_ghost"
    elif ghost_ratio < _STRONG_GHOST_RATIO or trough_z > _ISOLATED_TROUGH_ZSCORE:
        verdict = "strong_ghost"
    else:
        verdict = "weak_ghost"

    return {
        "ghost_quality": ghost_quality,
        "ghost_ratio": ghost_ratio,
        "curve_flatness": curve_flatness,
        "trough_z_score": trough_z,
        "verdict": verdict,
        "nse_min": min_nse,
        "nse_median": median_nse,
    }


def detect_jpeg_ghost(image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """
    Run JPEG Ghost detection on *image_bytes*.

    Scoring heuristic
    -----------------
    • no_ghost (synthetic / never JPEG compressed) → high AI score  (0.80)
    • strong_ghost at very high quality (>90)      → suspicious     (0.65)
    • strong_ghost at plausible quality (55–90)    → likely real    (0.25)
    • weak_ghost                                   → neutral        (0.50)

    Returns signal dict compatible with the ensemble detector.
    """
    _ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    _is_lossless = _ext in ("png", "gif", "bmp", "tiff", "tif", "webp")

    # Return neutral immediately for lossless formats — they have no prior
    # JPEG history, so the NSE curve is always flat and will be misinterpreted
    # as "no ghost = AI". Running the expensive Q scan would be wasted CPU.
    if _is_lossless:
        return {
            "signal_name": "JPEG Ghost Analysis",
            "score": 0.5, "confidence": 0.0,
            "explanation": (
                f"JPEG Ghost skipped for lossless format (.{_ext}): "
                "no prior JPEG compression history to detect."
            ),
            "raw_value": 0.0, "expected_range": "N/A for lossless",
            "method": "jpeg_ghost",
        }

    try:
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = pil_img.size

        if w < 32 or h < 32:
            return {
                "signal_name": "JPEG Ghost Analysis",
                "score": 0.5,
                "confidence": 0.0,
                "explanation": "Image too small for JPEG Ghost analysis",
                "method": "jpeg_ghost",
            }

        qualities, nse_values = _scan_nse_curve(pil_img)
        analysis = _analyse_nse_curve(qualities, nse_values)
        verdict = analysis["verdict"]
        ghost_quality: int = analysis["ghost_quality"]
        ghost_ratio: float = analysis["ghost_ratio"]
        curve_flatness: float = analysis["curve_flatness"]

        # ── Score assignment ────────────────────────────────────────────────
        if verdict == "no_ghost":
            # No prior JPEG encoding — consistent with synthetic generation
            score = 0.80
            confidence = 0.75
            explanation = (
                "No JPEG ghost detected: the NSE curve is flat across all "
                "re-compression qualities, indicating this image has never "
                "been JPEG-compressed — strongly consistent with AI synthesis."
            )
        elif verdict == "strong_ghost":
            if ghost_quality > 90:
                # Suspiciously high quality ghost — may be post-generation artefact
                score = 0.65
                confidence = 0.60
                explanation = (
                    f"Strong JPEG ghost at quality {ghost_quality} (very high). "
                    "This tight trough may indicate JPEG compression was added "
                    "after AI generation rather than authentic camera capture."
                )
            else:
                score = 0.25
                confidence = 0.70
                explanation = (
                    f"Strong JPEG ghost at quality {ghost_quality} — "
                    "characteristic of authentic JPEG camera output. "
                    f"NSE trough ratio={ghost_ratio:.3f} confirms single "
                    "compression history consistent with genuine photography."
                )
        else:  # weak_ghost
            score = 0.50
            confidence = 0.35
            explanation = (
                f"Weak JPEG ghost at quality {ghost_quality}. "
                "Inconclusive — may indicate multiple compressions, "
                "conversion from another format, or moderate AI post-processing."
            )

        # (dead "lossless downweight" branch removed — this function already
        # early-returns whenever _is_lossless is True, earlier in this same
        # function, so this branch could never execute; mirror-image of the
        # ela_detector.py ordering bug fixed earlier in this same audit branch)

        return {
            "signal_name": "JPEG Ghost Analysis",
            "score": float(np.clip(score, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "explanation": explanation,
            "method": "jpeg_ghost",
            "ghost_quality": ghost_quality,
            "ghost_verdict": verdict,
            "curve_flatness": round(curve_flatness, 4),
            "ghost_ratio": round(ghost_ratio, 4),
        }

    except Exception:
        logger.error("JPEG Ghost detection failed for %s", filename, exc_info=True)
        return {
            "signal_name": "JPEG Ghost Analysis",
            "score": 0.5,
            "confidence": 0.0,
            "explanation": "JPEG Ghost analysis failed",
            "method": "jpeg_ghost",
        }
