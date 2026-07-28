"""
Platt Scaling Confidence Calibration.

P(y=1 | f) = sigmoid(A * f + B)

Parameters are loaded once at module import and cached in memory.
Previously _load_params() was called on every calibrate() invocation,
causing 30 unnecessary file reads per image analysis (one per signal).

If no fitted parameters exist the module falls back to sensible defaults
(A = 5.0, B = -2.5) which approximate the prior hand-tuned curve.

Wilson score intervals
----------------------
For a calibrated probability p:
  centre = (p + z²/2n) / (1 + z²/n)
  half_w = z * sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)
where z = 1.645 (90% two-sided) and n = sum of signal confidences.

Persistence
-----------
Parameters saved to data/reference/platt_params.json.
Absent file → default parameters. Safe to deploy without retraining.
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.core.logger import setup_logger

logger = setup_logger(__name__)

_PARAMS_PATH = Path(__file__).parent.parent.parent / "data" / "reference" / "platt_params.json"

# Default parameters: sigmoid(A*f + B) is monotonically increasing in f.
# calibrate(0)≈0.08  calibrate(0.5)=0.50  calibrate(1)≈0.92
_DEFAULT_A = 5.0
_DEFAULT_B = -2.5

# Wilson score z-value for 90% two-sided interval
_WILSON_Z = 1.645

# ── Module-level param cache — loaded once, never re-read from disk ────────
_cached_A: Optional[float] = None
_cached_B: Optional[float] = None


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _load_params() -> Tuple[float, float]:
    """
    Return cached (A, B). Load from disk on first call, then cache in
    module-level variables so subsequent calls are pure memory reads.
    """
    global _cached_A, _cached_B
    if _cached_A is not None and _cached_B is not None:
        return _cached_A, _cached_B
    try:
        if _PARAMS_PATH.exists():
            data = json.loads(_PARAMS_PATH.read_text(encoding="utf-8"))
            _cached_A = float(data["A"])
            _cached_B = float(data["B"])
            logger.info("Loaded Platt params A=%.4f B=%.4f from %s", _cached_A, _cached_B, _PARAMS_PATH)
            return _cached_A, _cached_B
    except Exception as exc:
        logger.warning("Could not load Platt params: %s — using defaults", exc)
    _cached_A, _cached_B = _DEFAULT_A, _DEFAULT_B
    return _cached_A, _cached_B


def reload_params() -> Tuple[float, float]:
    """
    Force a re-read from disk (call after save_params() or after
    replacing platt_params.json without restarting the process).
    """
    global _cached_A, _cached_B
    _cached_A = None
    _cached_B = None
    return _load_params()


def save_params(A: float, B: float) -> None:
    """Persist fitted Platt parameters to disk and refresh in-memory cache."""
    global _cached_A, _cached_B
    _PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PARAMS_PATH.write_text(
        json.dumps({"A": A, "B": B}, indent=2), encoding="utf-8"
    )
    _cached_A, _cached_B = A, B
    logger.info("Saved Platt params A=%.4f B=%.4f to %s", A, B, _PARAMS_PATH)


def calibrate(raw_score: float) -> float:
    """Apply Platt scaling to a raw ensemble score → calibrated probability [0, 1]."""
    A, B = _load_params()
    return _sigmoid(A * raw_score + B)


def _wilson_interval(p: float, n_eff: float) -> tuple:
    """Shared Wilson score 90% interval math around a final probability p."""
    z = _WILSON_Z
    z2 = z * z
    centre = (p + z2 / (2 * n_eff)) / (1 + z2 / n_eff)
    half_w = (z * math.sqrt(p * (1 - p) / n_eff + z2 / (4 * n_eff * n_eff))) / (
        1 + z2 / n_eff
    )
    lower = max(0.0, centre - half_w)
    upper = min(1.0, centre + half_w)
    return lower, upper


def calibrate_with_interval(
    raw_score: float,
    signals: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Calibrate a raw score and compute Wilson score 90% confidence interval.

    Use only with a genuinely raw/uncalibrated ensemble score (i.e. a score
    that has NOT already had Platt scaling or an XGBoost predict_proba
    applied to it). For an already-final probability, use
    interval_around_calibrated() instead -- re-applying the sigmoid here to
    an already-calibrated input double-transforms it (F-3).

    Args:
        raw_score: Raw ensemble weighted sum.
        signals:   List of signal dicts with 'confidence' for effective n.

    Returns:
        {"calibrated": float, "interval_90": [lower, upper], "A": float, "B": float}
    """
    A, B = _load_params()
    p = _sigmoid(A * raw_score + B)

    n_eff = max(1.0, sum(s.get("confidence", 0.0) for s in signals)) if signals else 10.0
    lower, upper = _wilson_interval(p, n_eff)

    return {
        "calibrated":  round(p, 4),
        "interval_90": [round(lower, 4), round(upper, 4)],
        "A":           round(A, 4),
        "B":           round(B, 4),
    }


def interval_around_calibrated(
    calibrated_score: float,
    signals: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Compute a Wilson score 90% confidence interval around a probability
    that is ALREADY final (F-3) -- no further sigmoid/Platt transform is
    applied. This is what advanced_ensemble_detector.py's combine_signals()
    calls, because by the time it builds the calibration block,
    weighted_score has already been calibrated once (either via Platt in
    the fallback branch, or via XGBoost's own predict_proba). Passing that
    already-final score into calibrate_with_interval() re-applied the
    sigmoid a second time, so result["calibration"]["calibrated"] silently
    disagreed with the headline result["ai_probability"] -- the gap grew
    largest at the high-confidence extremes, which is exactly where a
    forensics tool's credibility matters most.

    Args:
        calibrated_score: An already-final probability in [0, 1].
        signals:          List of signal dicts with 'confidence' for effective n.

    Returns:
        {"calibrated": float, "interval_90": [lower, upper], "A": float, "B": float}
        "A"/"B" are still reported (current Platt params) for observability,
        even though they are not applied to calibrated_score itself.
    """
    A, B = _load_params()
    p = max(1e-6, min(1 - 1e-6, float(calibrated_score)))

    n_eff = max(1.0, sum(s.get("confidence", 0.0) for s in signals)) if signals else 10.0
    lower, upper = _wilson_interval(p, n_eff)

    return {
        "calibrated":  round(p, 4),
        "interval_90": [round(lower, 4), round(upper, 4)],
        "A":           round(A, 4),
        "B":           round(B, 4),
    }


def fit(
    raw_scores: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 200,
    lr: float = 0.01,
) -> Tuple[float, float]:
    """
    Fit Platt scaling parameters A, B by maximum likelihood (gradient descent).

    Args:
        raw_scores: 1-D array of raw ensemble scores.
        labels:     1-D binary array (1 = AI, 0 = real).
        max_iter:   Gradient descent iterations.
        lr:         Learning rate.

    Returns:
        (A, B) — also persisted to data/reference/platt_params.json.
    """
    scores = np.asarray(raw_scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    A = float(_DEFAULT_A)
    B = float(_DEFAULT_B)

    for _ in range(max_iter):
        logits = A * scores + B
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        error = probs - y
        dA = float(np.mean(error * scores))
        dB = float(np.mean(error))
        A -= lr * dA
        B -= lr * dB

    logger.info("Platt fit done: A=%.4f B=%.4f", A, B)
    save_params(A, B)
    return A, B
