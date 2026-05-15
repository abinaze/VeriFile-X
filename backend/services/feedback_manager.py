"""
Nash Equilibrium Adaptive Detection — Phase 29.

When an analyst marks a forensic result as incorrect, the signals that
were most responsible for the wrong prediction receive a weight penalty.
Over time, consistently misleading signals converge to lower weights —
a Nash equilibrium where no single signal benefits from further change.

Storage
-------
data/feedback.jsonl         — append-only analyst feedback log
data/signal_weights.json    — current adaptive weight overrides

Algorithm (gradient-based Nash update)
---------------------------------------
For each feedback record:
  1. Identify which signals were "guilty" — score agreed with the wrong
     prediction (e.g. signal said AI, true label is real).
  2. Apply a penalty: weight_i = weight_i * (1 - lr * |score_i - label|)
  3. Clip weights to [0.05, 2.0] to prevent starvation.
  4. Normalise so weights sum to 1.0.
  5. Persist updated weights to signal_weights.json.

The weights stored here are *multipliers* on top of the ensemble's
baseline weights — not absolute values. A weight of 1.0 = no change.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR        = Path(__file__).parent.parent.parent / "data"
_FEEDBACK_PATH   = _DATA_DIR / "feedback.jsonl"
_WEIGHTS_PATH    = _DATA_DIR / "signal_weights.json"

_write_lock = threading.Lock()

_LEARNING_RATE   = 0.05
_MIN_WEIGHT      = 0.05
_MAX_WEIGHT      = 2.00
_GUILTY_THRESHOLD = 0.3   # signal must agree with wrong prediction by this much


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Weight persistence ────────────────────────────────────────────────────────

def load_weights() -> Dict[str, float]:
    """Load current signal weight multipliers. Returns {} if file absent."""
    try:
        if _WEIGHTS_PATH.exists():
            data = json.loads(_WEIGHTS_PATH.read_text(encoding="utf-8"))
            return {k: float(v) for k, v in data.items()}
    except Exception as exc:
        logger.warning("Could not load signal weights: %s", exc)
    return {}


def _save_weights(weights: Dict[str, float]) -> None:
    _ensure_dirs()
    with _write_lock:
        _WEIGHTS_PATH.write_text(
            json.dumps(weights, indent=2, sort_keys=True), encoding="utf-8"
        )


# ── Feedback recording ────────────────────────────────────────────────────────

def record_feedback(
    evidence_id: str,
    true_label: str,          # "ai_generated" | "authentic"
    predicted_label: str,
    signals: List[Dict[str, Any]],
    analyst_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record analyst feedback and update signal weights.

    Args:
        evidence_id:      UUID of the evidence being corrected.
        true_label:       Ground truth — "ai_generated" or "authentic".
        predicted_label:  What the system said.
        signals:          List of signal dicts with 'signal_name' and 'score'.
        analyst_notes:    Optional free-text from the analyst.

    Returns:
        {feedback_id, updated_signals, weight_delta_summary}
    """
    if true_label not in ("ai_generated", "authentic"):
        raise ValueError("true_label must be 'ai_generated' or 'authentic'")

    true_ai = 1.0 if true_label == "ai_generated" else 0.0
    was_wrong = true_label != predicted_label

    feedback_id = f"fb-{evidence_id[:8]}-{int(datetime.now().timestamp())}"

    # Load current weights
    weights = load_weights()

    updated: List[str] = []
    if was_wrong and signals:
        for sig in signals:
            name  = sig.get("signal_name", "")
            score = float(sig.get("score", 0.5))

            # "Guilty" = signal strongly agreed with the wrong prediction.
            # error = |score - true_label| measures how wrong the signal was.
            # High error means the signal pointed away from truth (guilty).
            # Low error means the signal was actually correct — skip it.
            error = abs(score - true_ai)
            if error <= _GUILTY_THRESHOLD:
                continue  # Signal was close to correct — don't penalise

            guilt = error   # how strongly it agreed with wrong answer
            current = weights.get(name, 1.0)
            new_w   = current * (1.0 - _LEARNING_RATE * guilt)
            new_w   = max(_MIN_WEIGHT, min(_MAX_WEIGHT, new_w))
            weights[name] = round(new_w, 6)
            updated.append(f"{name}: {current:.4f} → {new_w:.4f}")

        _save_weights(weights)

    # Append to feedback log
    record = {
        "feedback_id":     feedback_id,
        "evidence_id":     evidence_id,
        "true_label":      true_label,
        "predicted_label": predicted_label,
        "was_wrong":       was_wrong,
        "analyst_notes":   analyst_notes,
        "signals_count":   len(signals),
        "weights_updated": updated,
        "timestamp":       _now(),
    }

    _ensure_dirs()
    with _write_lock:
        with _FEEDBACK_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    logger.info(
        "Feedback recorded: %s true=%s predicted=%s wrong=%s updated=%d weights",
        evidence_id, true_label, predicted_label, was_wrong, len(updated),
    )

    return {
        "feedback_id":       feedback_id,
        "was_wrong":         was_wrong,
        "weights_updated":   updated,
        "total_signals":     len(signals),
    }


def get_feedback_history(
    evidence_id: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return recent feedback records, optionally filtered by evidence_id."""
    if not _FEEDBACK_PATH.exists():
        return []
    try:
        lines = _FEEDBACK_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    results = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evidence_id and rec.get("evidence_id") != evidence_id:
            continue
        results.append(rec)
        if len(results) >= limit:
            break
    return results


def get_weight_summary() -> Dict[str, Any]:
    """Return current adaptive weight multipliers and metadata."""
    weights = load_weights()
    return {
        "signal_weights":  weights,
        "total_overrides": len(weights),
        "signals_penalised": sum(1 for v in weights.values() if v < 1.0),
        "signals_boosted":   sum(1 for v in weights.values() if v > 1.0),
    }
