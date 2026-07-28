import pickle
from backend.core.logger import setup_logger
import numpy as np
from pathlib import Path
from typing import Dict, Any

logger = setup_logger(__name__)

CENTROIDS_PATH = Path(__file__).parent.parent.parent / "data" / "reference" / "own_centroids.pkl"

# BUG FIX: this detector previously never used the shared ModelCache
# singleton (backend/core/model_cache.py) that dire_detector.py and
# clip_detector.py both correctly use. Combined with a fresh
# OwnEmbeddingDetector() being constructed on EVERY analysis request
# (AdvancedEnsembleDetector.__init__ does self.own_detector =
# OwnEmbeddingDetector()), the instance-scoped _model_loaded flag provided
# zero benefit across requests — every single request reloaded the full
# model from disk. Now cached under this key, same pattern as its siblings.
_CACHE_KEY = "own-embedding-model"


class OwnEmbeddingDetector:

    def __init__(self):
        self.device           = None
        self.model            = None
        self._model_loaded    = False
        self.real_centroid    = None
        self.ai_centroid      = None
        self._centroid_loaded = False
        from backend.core.model_cache import get_model_cache
        self.cache = get_model_cache()

    def _load_model(self):
        if self._model_loaded:
            return
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        cached = self.cache.get(_CACHE_KEY)
        if cached is not None:
            self.model = cached
            self._model_loaded = True
            logger.info("OwnEmbeddingDetector: reused cached model (no reload)")
            return

        # Cache miss - load from disk. Guarded by a per-key lock (F-7):
        # without it, two concurrent cold-start requests can both observe
        # the miss above and both independently load the model.
        with self.cache.get_load_lock(_CACHE_KEY):
            # Double-check: another thread may have finished loading and
            # populated the cache while we were waiting for the lock.
            cached = self.cache.get(_CACHE_KEY)
            if cached is not None:
                self.model = cached
                self._model_loaded = True
                logger.info("OwnEmbeddingDetector: reused cached model (loaded by a concurrent request)")
                return

            from backend.services.own_detector.model import load_model
            self.model  = load_model(self.device)
            self._model_loaded = True
            if self.model is not None:
                self.model.eval()
                # Estimate ~20MB for the trained EfficientNet-B0 checkpoint —
                # matches the ballpark used in config/cache_config.py's
                # MODEL_SIZES table for similarly-sized models.
                self.cache.set(_CACHE_KEY, self.model, size_mb=20)
                logger.info(f"OwnEmbeddingDetector loaded on {self.device} and cached")
            else:
                logger.warning("OwnEmbeddingDetector: no trained model found, signal will return neutral 0.5")

    def _load_centroids(self):
        if self._centroid_loaded:
            return
        if not CENTROIDS_PATH.exists():
            logger.warning("own_centroids.pkl not found. Run scripts/build_centroids.py first.")
            self.real_centroid    = None
            self.ai_centroid      = None
            self._centroid_loaded = True
            return
        try:
            with open(CENTROIDS_PATH, "rb") as f:
                db = pickle.load(f)
            self.real_centroid = db["real_centroid"]
            self.ai_centroid   = db["ai_centroid"]
            logger.info(
                "Loaded centroids: %d real, %d AI, sep=%.4f",
                db["real_count"], db["ai_count"], db["separation"],
            )
        except Exception as exc:
            # Handles Git LFS pointer stubs (tiny ASCII text) and corrupt files.
            # Always set _centroid_loaded=True so we don't re-attempt on every
            # request — the file isn't going to fix itself at runtime.
            logger.error(
                "own_centroids.pkl could not be loaded (%s). "
                "Likely a Git LFS pointer stub — run: git lfs pull. "
                "Falling back to direct-only embedding (no centroid scoring).",
                exc,
            )
            self.real_centroid = None
            self.ai_centroid   = None
        finally:
            self._centroid_loaded = True

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _neutral_result(self, reason: str) -> Dict[str, Any]:
        return {
            "signal_name":    "Own Embedding Detection",
            "score":          0.5,
            "confidence":     0.0,
            "explanation":    f"Skipped: {reason}. Run scripts/train_embedding.py first.",
            "raw_value":      0.5,
            "expected_range": "> 0.5 for AI",
            "method":         "own_embedding",
        }

    def detect(self, image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
        try:
            self._load_model()
            self._load_centroids()

            if self.model is None:
                return self._neutral_result("no trained model file found")

            import torch
            from PIL import Image
            from io import BytesIO
            from backend.services.own_detector.model import TRANSFORM

            img    = Image.open(BytesIO(image_bytes)).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding, direct_prob = self.model(tensor)

            embedding_np = embedding.cpu().numpy().squeeze()
            embedding_np = embedding_np / (np.linalg.norm(embedding_np) + 1e-8)
            direct_score = float(direct_prob.item())

            if self.real_centroid is not None and self.ai_centroid is not None:
                sim_real       = self._cosine_similarity(embedding_np, self.real_centroid)
                sim_ai         = self._cosine_similarity(embedding_np, self.ai_centroid)
                exp_ai         = np.exp(sim_ai * 10)
                exp_real       = np.exp(sim_real * 10)
                centroid_score = float(exp_ai / (exp_ai + exp_real))
                ai_score       = 0.6 * direct_score + 0.4 * centroid_score
                confidence     = 0.85
                method         = "efficientnet_direct+centroid"
            else:
                ai_score   = direct_score
                confidence = 0.70
                method     = "efficientnet_direct_only"

            if ai_score > 0.7:
                explanation = f"Embedding strongly matches AI-generated image patterns (score={ai_score:.3f})"
            elif ai_score > 0.5:
                explanation = f"Embedding leans toward AI-generated patterns (score={ai_score:.3f})"
            elif ai_score > 0.3:
                explanation = f"Embedding leans toward authentic image patterns (score={ai_score:.3f})"
            else:
                explanation = f"Embedding strongly matches authentic image patterns (score={ai_score:.3f})"

            return {
                "signal_name":    "Own Embedding Detection",
                "score":          float(ai_score),
                "confidence":     confidence,
                "explanation":    explanation,
                "raw_value":      round(ai_score, 4),
                "expected_range": "> 0.5 for AI",
                "method":         method,
            }

        except Exception as e:
            logger.warning(f"OwnEmbeddingDetector failed: {e}")
            return self._neutral_result(f"exception: {str(e)}")

    def cleanup(self):
        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()
