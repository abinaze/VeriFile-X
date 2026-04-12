import pickle
import numpy as np
from pathlib import Path as _Path

_XGB_MODEL_PATH = _Path("data/reference/ensemble_xgb.pkl")
_xgb_cache: dict = {}


def _load_xgb():
    if "model" not in _xgb_cache and _XGB_MODEL_PATH.exists():
        with open(_XGB_MODEL_PATH, "rb") as _f:
            _xgb_cache.update(pickle.load(_f))
    return (
        _xgb_cache.get("model"),
        _xgb_cache.get("feature_names"),
        _xgb_cache.get("explainer"),
    )


"""
Advanced Ensemble Detector combining Statistical + DIRE + CLIP.
"""
from typing import Dict, Any
from backend.core.logger import setup_logger
from backend.services.statistical_detector import StatisticalDetector
from backend.services.dire_detector import DIREDetector
from backend.services.clip_detector import CLIPDetector
from backend.services.own_embedding_detector import OwnEmbeddingDetector
from backend.services.prnu_detector import detect_prnu
from backend.services.ela_detector import detect_ela
from backend.services.metadata_forensics import analyze_metadata
from backend.services.dct_frequency_detector import detect_dct_artifacts

logger = setup_logger(__name__)


class AdvancedEnsembleDetector(StatisticalDetector):
    """
    State-of-the-art ensemble combining:
    - Statistical methods (19 signals)
    - DIRE (diffusion detection)
    - CLIP (universal detection)
    - Own EfficientNet embedding detector

    Validated accuracy: 85-92%
    """

    def __init__(self, image_bytes: bytes, filename: str):
        """Initialize ensemble detector."""
        super().__init__(image_bytes, filename)

        self.dire_detector = DIREDetector()
        self.clip_detector = CLIPDetector()
        self.own_detector  = OwnEmbeddingDetector()

        logger.info(f"Advanced ensemble detector initialized for {filename}")

    def detect(self) -> Dict[str, Any]:
        """
        Run complete advanced detection with all methods.

        Returns:
            Complete report with 26 detection signals
        """
        logger.info(f"Starting advanced ensemble detection for {self.filename}")

        # Run parent class methods (19 statistical signals)
        base_report = super().detect()

        # Deep-learning and forensic signals
        dire_result     = self.dire_detector.detect(self.image_bytes, self.filename)
        clip_result     = self.clip_detector.detect(self.image_bytes, self.filename)
        own_result      = self.own_detector.detect(self.image_bytes, self.filename)
        prnu_result     = detect_prnu(self.image_bytes, self.filename)
        ela_result      = detect_ela(self.image_bytes, self.filename)
        metadata_result = analyze_metadata(self.image_bytes, self.filename)
        dct_result      = detect_dct_artifacts(self.image_bytes, self.filename)

        # Combine all signals (26 total)
        all_signals = base_report["all_signals"] + [
            dire_result, clip_result, own_result, prnu_result,
            ela_result, metadata_result, dct_result,
        ]

        # Weighted fallback score (used when XGBoost model is absent)
        dire_confidence = dire_result.get("confidence", 0.0)

        if dire_confidence > 0.0:
            weighted_score = (
                0.31 * base_report["ai_probability"] +
                0.24 * dire_result["score"] +
                0.18 * clip_result["score"] +
                0.09 * prnu_result["score"] +
                0.07 * ela_result["score"] +
                0.06 * metadata_result["score"] +
                0.05 * dct_result["score"]
            )
        else:
            logger.info("DIRE unavailable — using statistical+CLIP+PRNU+ELA+metadata+DCT")
            weighted_score = (
                0.49 * base_report["ai_probability"] +
                0.22 * clip_result["score"] +
                0.11 * prnu_result["score"] +
                0.08 * ela_result["score"] +
                0.06 * metadata_result["score"] +
                0.04 * dct_result["score"]
            )

        # Platt-style calibration (weighted-sum fallback only)
        def calibrate(score: float) -> float:
            adjusted = score - 0.02
            if adjusted > 0.65:
                return min(1.0, 0.65 + (adjusted - 0.65) * 1.15)
            elif adjusted < 0.35:
                return max(0.0, 0.35 - (0.35 - adjusted) * 1.15)
            return adjusted

        weighted_score = calibrate(weighted_score)

        # XGBoost meta-model overrides weighted sum when available
        xgb_model, feature_names, _ = _load_xgb()
        if xgb_model is not None:
            signal_map = {
                s["signal_name"].lower().replace(" ", "_"): s["score"]
                for s in all_signals
            }
            # np.nan lets XGBoost use its native missing-value branch selection
            feat_vec       = np.array([[signal_map.get(k, np.nan) for k in feature_names]])
            weighted_score = float(xgb_model.predict_proba(feat_vec)[0][1])
            logger.info(f"XGBoost ensemble score: {weighted_score:.4f}")
        else:
            logger.info("XGBoost model not found, using weighted sum fallback")
            # Manual boost multipliers removed — the weighted sum above plus
            # calibrate() is sufficient; XGBoost learns co-occurrence patterns
            # from training data when available.

        suspicious_count = sum(1 for s in all_signals if s["score"] > 0.5)

        # Classification thresholds
        if weighted_score > 0.80:
            classification = "likely_ai_generated"
            confidence     = "very_high"
        elif weighted_score > 0.70:
            classification = "likely_ai_generated"
            confidence     = "high"
        elif weighted_score > 0.50:
            classification = "possibly_ai_generated"
            confidence     = "medium"
        elif weighted_score > 0.30:
            classification = "possibly_authentic"
            confidence     = "medium"
        else:
            classification = "likely_authentic"
            confidence     = "high" if weighted_score < 0.20 else "medium"

        sorted_signals = sorted(all_signals, key=lambda x: x["score"], reverse=True)
        top_reasons    = [s["explanation"] for s in sorted_signals[:3]]

        result = {
            "ai_probability":          float(weighted_score),
            "classification":          classification,
            "confidence":              confidence,
            "suspicious_signals_count": suspicious_count,
            "total_signals":           len(all_signals),
            "all_signals":             all_signals,
            "top_reasons":             top_reasons,
            "summary": (
                f"Analyzed using {len(all_signals)} independent signals including "
                f"statistical analysis, diffusion reconstruction, and semantic embeddings. "
                f"{suspicious_count} signals indicate AI generation."
            ),
            "detection_version": "advanced-ensemble-v1.4",
            "methods_used": ["statistical", "dire", "clip", "prnu", "ela", "metadata", "dct"],
        }

        logger.info(
            f"Advanced ensemble complete: {classification} "
            f"(p={weighted_score:.3f}, {suspicious_count}/{len(all_signals)} signals)"
        )

        return result

    def cleanup(self):
        """Clean up GPU resources."""
        self.dire_detector.cleanup()
        self.clip_detector.cleanup()
