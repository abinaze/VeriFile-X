"""
Advanced Ensemble Detector combining Statistical + DIRE + CLIP + Phase-20/21/22 signals.

This module must have its docstring as the first statement so that
help(), IDEs, and documentation generators can find it.
"""
import pickle
import numpy as np
from pathlib import Path as _Path

_XGB_MODEL_PATH = _Path(__file__).parent.parent.parent / "data" / "reference" / "ensemble_xgb.pkl"
_xgb_cache: dict = {}


def _load_xgb():
    if "model" not in _xgb_cache and _XGB_MODEL_PATH.exists():
        try:
            from backend.core.model_integrity import verify_integrity, ModelIntegrityError
            verify_integrity(_XGB_MODEL_PATH)
            with open(_XGB_MODEL_PATH, "rb") as _f:
                _xgb_cache.update(pickle.load(_f))
        except ModelIntegrityError:
            # A hash mismatch is a materially different, higher-severity
            # signal than "file missing/corrupt" -- let it propagate
            # rather than silently falling back to the weighted-sum
            # ensemble as if nothing were wrong.
            raise
        except Exception as _e:
            # Corrupt file or Git LFS pointer stub — log and skip gracefully
            import logging as _log
            _log.getLogger(__name__).warning(
                f"XGBoost model could not be loaded ({_e}). "
                "Falling back to weighted sum ensemble."
            )
    return (
        _xgb_cache.get("model"),
        _xgb_cache.get("feature_names"),
        _xgb_cache.get("explainer"),
    )

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
from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
from backend.services.noise_map_detector import detect_noise_map
from backend.services.noiseprint_detector import detect_noiseprint
from backend.services.cfa_detector import detect_cfa_artifacts

logger = setup_logger(__name__)


def _aggregate_stat_confidence(all_sub_signals: list) -> float:
    """Aggregate confidence for the 19-signal statistical bundle (F-13).

    Previously hardcoded to 1.0 regardless of how many of the 19
    sub-signals actually succeeded. Each sub-signal already reports its
    own real confidence (e.g. 0.92 on success, a 0.3 fallback on
    "Analysis failed - insufficient data") -- averaging those gives the
    outer ensemble something real to gate on, instead of always fully
    trusting the bundle.
    """
    if not all_sub_signals:
        return 0.0
    return sum(s.get("confidence", 0.0) for s in all_sub_signals) / len(all_sub_signals)


class AdvancedEnsembleDetector(StatisticalDetector):
    """
    State-of-the-art ensemble combining:
    - Statistical methods (19 signals)
    - DIRE (diffusion detection)
    - CLIP (universal detection)
    - Own EfficientNet embedding detector

    No accuracy percentage is published here deliberately -- see
    README.md's "Accuracy, Validation, and Honest Limitations" section.
    This docstring previously claimed "Validated accuracy: 85-92%", the
    same unverified figure already removed from the README and frontend
    marketing copy; it was simply never propagated to this file.
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
            Complete report with 30 detection signals
        """
        logger.info(f"Starting advanced ensemble detection for {self.filename}")

        # Run parent class methods (19 statistical signals)
        base_report = super().detect()

        # Gate camera-forensic signals by image content type.
        # PRNU/ELA/metadata are designed for camera photos — running them on
        # screenshots, illustrations, or documents injects noise into the ensemble.
        from backend.services.image_type_classifier import classify_image_type as _classify
        _img_type = _classify(self.image_bytes, self.filename)
        logger.info(
            "Image type: %s (confidence=%.2f) run_prnu=%s run_ela=%s run_metadata=%s",
            _img_type["image_type"], _img_type["confidence"],
            _img_type["run_prnu"], _img_type["run_ela"], _img_type["run_metadata"],
        )

        def _skipped(signal_name: str, method: str) -> dict:
            return {
                "signal_name":    signal_name,
                "score":          0.5,
                "confidence":     0.0,
                "explanation":    (
                    f"Skipped — not appropriate for {_img_type['image_type']} images "
                    f"({_img_type['explanation']})"
                ),
                "raw_value":      0.5,
                "expected_range": "> 0.5 for AI",
                "method":         method,
            }

        # Deep-learning and forensic signals
        dire_result       = self.dire_detector.detect(self.image_bytes, self.filename)
        clip_result       = self.clip_detector.detect(self.image_bytes, self.filename)
        own_result        = self.own_detector.detect(self.image_bytes, self.filename)
        prnu_result       = (detect_prnu(self.image_bytes, self.filename)
                             if _img_type["run_prnu"]
                             else _skipped("PRNU Camera Fingerprint", "prnu_skipped"))
        ela_result        = (detect_ela(self.image_bytes, self.filename)
                             if _img_type["run_ela"]
                             else _skipped("ELA Compression Analysis", "ela_skipped"))
        metadata_result   = (analyze_metadata(self.image_bytes, self.filename)
                             if _img_type["run_metadata"]
                             else _skipped("Metadata Forensics", "metadata_skipped"))
        dct_result        = detect_dct_artifacts(self.image_bytes, self.filename)
        jpeg_ghost_result = detect_jpeg_ghost(self.image_bytes, self.filename)
        noise_map_result  = detect_noise_map(self.image_bytes, self.filename)
        noiseprint_result = detect_noiseprint(self.image_bytes, self.filename)
        cfa_result        = detect_cfa_artifacts(self.image_bytes, self.filename)

        return self.combine_signals(
            base_report, dire_result, clip_result, own_result, prnu_result,
            ela_result, metadata_result, dct_result, jpeg_ghost_result,
            noise_map_result, noiseprint_result, cfa_result,
        )

    def combine_signals(
        self,
        base_report:       Dict[str, Any],
        dire_result:       Dict[str, Any],
        clip_result:       Dict[str, Any],
        own_result:        Dict[str, Any],
        prnu_result:       Dict[str, Any],
        ela_result:        Dict[str, Any],
        metadata_result:   Dict[str, Any],
        dct_result:        Dict[str, Any],
        jpeg_ghost_result: Dict[str, Any],
        noise_map_result:  Dict[str, Any],
        noiseprint_result: Dict[str, Any],
        cfa_result:        Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine pre-computed signal results into the final ensemble report.

        Extracted from detect() so callers that have already run each
        detector individually (e.g. the SSE streaming analyser) can reuse
        the exact same scoring / calibration / classification logic without
        executing any detector a second time.
        """
        # Combine all 30 signals
        all_signals = base_report["all_signals"] + [
            dire_result, clip_result, own_result, prnu_result,
            ela_result, metadata_result, dct_result,
            jpeg_ghost_result, noise_map_result,
            noiseprint_result, cfa_result,
        ]

        # Weighted fallback score (used when XGBoost model is absent).
        # Use explicit "available" flag rather than inferring from confidence.
        # A failed DIRE that returns a small fallback confidence (e.g. 0.1)
        # would otherwise trigger the high-weight DIRE branch incorrectly.
        dire_available = bool(dire_result.get("available", False))

        # UNIFIED SCORING PATH.
        # Previously this branched into two independently-maintained
        # formulas: a DIRE-available path with hardcoded static weights and
        # NO confidence gating (any inactive signal's neutral 0.5 was
        # blended in at full weight), and a DIRE-unavailable fallback path
        # with confidence gating AND analyst-feedback weight multipliers.
        # Every subsequent improvement to "how signals combine" landed only
        # in the fallback path — including confidence gating itself and the
        # feedback loop — meaning the recommended, higher-accuracy DIRE
        # deployment never received either improvement.
        #
        # Both refinements are strictly more correct in every case, so the
        # branch split is removed entirely: one signal list, one confidence
        # filter, one feedback-weighted renormalization, whether or not
        # DIRE happens to be available. DIRE/own_result simply drop out of
        # the active set (like any other signal) when their own confidence
        # is 0.0 (DIRE unavailable, or own_embedding model missing).
        from backend.services.feedback_manager import load_weights as _load_fb_weights
        _fb = _load_fb_weights()
        if _fb:
            logger.info("Applying %d analyst feedback weight overrides", len(_fb))

        def _fw(signal_name: str, base_weight: float) -> float:
            """Return base_weight * feedback multiplier, clamped to [0.05, 2.0]."""
            m = _fb.get(signal_name.lower(), 1.0)
            return base_weight * max(0.05, min(2.0, m))

        # Base weights sum to 1.00 across all 12 top-level ensemble inputs.
        # F-13: the top-level ensemble excludes any of the 11 named
        # signals when their confidence == 0 (see "_active" filter below),
        # but the "stat" bundle folding in all 19 statistical sub-signals
        # used to be hardcoded to confidence=1.0 regardless of how many
        # of those 19 actually succeeded. Each sub-signal method already
        # reports its own real confidence (e.g. 0.92 on success, a 0.3
        # fallback on "Analysis failed - insufficient data") -- this
        # averages those into one aggregate confidence for the bundle, so
        # a bad edge case in several sub-signals now measurably reduces
        # how much the outer ensemble trusts the bundle as a whole,
        # instead of it always being fully trusted.
        _stat_confidence = _aggregate_stat_confidence(base_report.get("all_signals", []))

        _raw_signals = [
            ("dire",       _fw("dire reconstruction error",   0.21), dire_result["score"],       dire_result.get("confidence", 0)),
            ("stat",       _fw("statistical analysis",        0.20), base_report["ai_probability"], _stat_confidence),
            ("clip",       _fw("clip embedding analysis",      0.14), clip_result["score"],       clip_result.get("confidence", 0)),
            ("own",        _fw("own embedding detection",      0.08), own_result["score"],       own_result.get("confidence", 0)),
            ("prnu",       _fw("prnu camera fingerprint",       0.08), prnu_result["score"],       prnu_result.get("confidence", 0)),
            ("ela",        _fw("ela compression analysis",      0.07), ela_result["score"],       ela_result.get("confidence", 0)),
            ("meta",       _fw("metadata forensics",             0.06), metadata_result["score"],  metadata_result.get("confidence", 0)),
            ("dct",        _fw("dct frequency artifacts",        0.05), dct_result["score"],       dct_result.get("confidence", 0)),
            ("jpeg_ghost", _fw("jpeg ghost analysis",             0.04), jpeg_ghost_result["score"], jpeg_ghost_result.get("confidence", 0)),
            ("noiseprint", _fw("noiseprint camera fingerprint",   0.03), noiseprint_result["score"], noiseprint_result.get("confidence", 0)),
            ("noise_map",  _fw("noise map analysis",              0.02), noise_map_result["score"], noise_map_result.get("confidence", 0)),
            ("cfa",        _fw("cfa artifact analysis",           0.02), cfa_result["score"],       cfa_result.get("confidence", 0)),
        ]
        # Filter: only include signals with confidence > 0, renormalize
        # remaining weights so they still sum to 1.0. This is the fix for
        # the "0.5 pollution" bug — a missing CLIP database, ELA skipped on
        # a lossless format, PRNU/ELA/metadata skipped for a non-camera
        # image type, etc. no longer drag the score toward 0.5 at full
        # static weight; they're excluded and everyone else is rescaled.
        _active = [(name, w, score) for name, w, score, conf in _raw_signals if conf > 0]
        if not _active:
            logger.warning("No active signals — returning neutral 0.5")
            weighted_score = 0.5
        else:
            _total = sum(w for _, w, _ in _active)
            weighted_score = sum((w / _total) * score for _, w, score in _active)
            logger.info(
                "Unified ensemble: %d/%d signals active (dire_available=%s), sum_w=%.4f",
                len(_active), len(_raw_signals), dire_available, _total,
            )

        # XGBoost meta-model overrides weighted sum when available.
        # NOTE: Platt calibration is applied ONLY on the fallback path below.
        # XGBoost.predict_proba already outputs calibrated probabilities;
        # applying Platt on top would distort those scores.
        xgb_model, feature_names, _ = _load_xgb()
        if xgb_model is not None:
            signal_map = {
                s["signal_name"].lower().replace(" ", "_"): s["score"]
                for s in all_signals
            }
            # Feature-name mismatch check: any XGBoost feature key not in
            # signal_map will be filled with np.nan (XGBoost native missing).
            # Mismatches are silent score degraders — log them explicitly.
            _missing = [k for k in feature_names if k not in signal_map]
            if _missing:
                logger.warning(
                    "XGBoost feature-name mismatch: %d/%d features not in live "
                    "signal_map (will use np.nan). Missing: %s. "
                    "Retrain ensemble_xgb.pkl or check signal_name strings.",
                    len(_missing), len(feature_names), _missing,
                )
            # np.nan lets XGBoost use its native missing-value branch selection
            feat_vec       = np.array([[signal_map.get(k, np.nan) for k in feature_names]])
            weighted_score = float(xgb_model.predict_proba(feat_vec)[0][1])
            logger.info(f"XGBoost ensemble score: {weighted_score:.4f}")
        else:
            logger.info("XGBoost model not found — applying Platt calibration to weighted sum")
            # Platt scaling calibration — proper sigmoid fit replacing hand-tuned stub.
            # Only reaches here when XGBoost is unavailable; when XGBoost IS available
            # its predict_proba output is already a calibrated probability.
            from backend.services.platt_calibrator import calibrate as _platt_calibrate
            weighted_score = _platt_calibrate(weighted_score)

        suspicious_count = sum(1 for s in all_signals if s["score"] > 0.5)

        # ── Classification thresholds ──────────────────────────────────────────
        # These are empirical and should be updated by running a ROC curve
        # analysis on a held-out benchmark after retraining. Current values
        # are conservative estimates that err toward "inconclusive" in the
        # 0.40–0.60 range to avoid overconfident wrong predictions when
        # forensic signals are weak or mixed.
        #
        # Updating them: fit Platt calibration on labeled data, then pick
        # thresholds that maximise F1 at your required precision/recall.
        _T_VERY_HIGH_AI  = 0.80   # > → "likely_ai_generated" / very_high confidence
        _T_HIGH_AI       = 0.70   # > → "likely_ai_generated" / high
        _T_MEDIUM_AI     = 0.60   # > → "possibly_ai_generated" / medium
        _T_INCONCLUSIVE  = 0.40   # ≥ → "inconclusive" / low
        _T_MEDIUM_REAL   = 0.30   # > → "possibly_authentic" / medium

        if weighted_score > _T_VERY_HIGH_AI:
            classification = "likely_ai_generated"
            confidence     = "very_high"
        elif weighted_score > _T_HIGH_AI:
            classification = "likely_ai_generated"
            confidence     = "high"
        elif weighted_score > _T_MEDIUM_AI:
            classification = "possibly_ai_generated"
            confidence     = "medium"
        elif weighted_score >= _T_INCONCLUSIVE:
            # Explicit inconclusive zone — conflicting or weak signals
            classification = "inconclusive"
            confidence     = "low"
        elif weighted_score > _T_MEDIUM_REAL:
            classification = "possibly_authentic"
            confidence     = "medium"
        else:
            classification = "likely_authentic"
            confidence     = "high" if weighted_score < 0.20 else "medium"

        sorted_signals = sorted(all_signals, key=lambda x: x["score"], reverse=True)
        # Filter out neutral placeholder signals (confidence=0 means signal was
        # unavailable/skipped, e.g. "CLIP database not built"). Including them
        # as top reasons is misleading — they contain no forensic evidence.
        top_reasons = [
            s["explanation"]
            for s in sorted_signals
            if s.get("confidence", 0.0) > 0.0
        ][:3]

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
                f"statistical analysis, diffusion reconstruction, semantic embeddings, "
                f"JPEG ghost analysis, noise map, Noiseprint, and CFA forensics. "
                f"{suspicious_count} signals indicate AI generation."
            ),
            "detection_version": "advanced-ensemble-v1.6",
            "methods_used": [
                "statistical", "dire", "clip", "own_embedding",
                "prnu", "ela", "metadata", "dct",
                "jpeg_ghost", "noise_map", "noiseprint", "cfa",
            ],
        }

        # MCMC probabilistic distribution
        from backend.services.mcmc_engine import run_mcmc as _run_mcmc
        import hashlib as _hl
        # Use the full image sha256 for the MCMC seed so that two images
        # with identical EXIF headers (same first 64 bytes) still get different
        # chains. This doesn't make MCMC non-deterministic per image — it still
        # retraces the same path each time the same image is analyzed (cached)
        # — but it prevents cross-image seed collisions.
        _seed = int(_hl.sha256(self.image_bytes).hexdigest()[:8], 16) % (2**31)
        probability_distribution = _run_mcmc(
            signals=all_signals,
            point_estimate=weighted_score,
            rng_seed=_seed,
        )
        result["probability_distribution"] = probability_distribution

        # Wilson interval around the already-calibrated score (F-3).
        # weighted_score is already final here -- either XGBoost's own
        # predict_proba (xgb branch) or a single Platt application
        # (fallback branch, above) -- so re-running it through Platt's
        # sigmoid a second time would double-transform it, causing
        # result["calibration"]["calibrated"] to silently disagree with
        # the headline result["ai_probability"] for the same report.
        from backend.services.platt_calibrator import interval_around_calibrated as _iac
        result["calibration"] = _iac(weighted_score, signals=all_signals)

        logger.info(
            f"Advanced ensemble complete: {classification} "
            f"(p={weighted_score:.3f}, certainty={probability_distribution['certainty']}, "
            f"{suspicious_count}/{len(all_signals)} signals)"
        )

        return result

    def cleanup(self):
        """Clean up GPU resources."""
        self.dire_detector.cleanup()
        self.clip_detector.cleanup()
