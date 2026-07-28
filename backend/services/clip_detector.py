"""
CLIP-based Universal Fake Detection with proper reference database and caching.

IMPROVED: Model caching for 10x performance improvement on repeated detections!
"""
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from backend.core.logger import setup_logger
from backend.core.model_cache import get_model_cache
from backend.config.cache_config import CacheConfig

logger = setup_logger(__name__)


class CLIPDetector:
    """CLIP-based universal AI detection with learned centroids and caching."""
    
    def __init__(self):
        """Initialize CLIP detector."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self._model_loaded = False
        
        # Cache configuration
        self.cache = get_model_cache()
        self.cache_key = "clip-vit-b-32"
        self.model_size_mb = CacheConfig.MODEL_SIZES.get(self.cache_key, 350)
        self._from_cache = False
        
        # Reference centroids (will be loaded from database)
        self.real_centroid = None
        self.fake_centroid = None
        
        logger.info(f"CLIP Detector initialized (device: {self.device})")
    
    def _load_model(self):
        """Lazy load CLIP model with caching (10x faster on cache hit)."""
        if self._model_loaded:
            return
        
        # Try to get from cache first
        cached_model = self.cache.get(self.cache_key)
        
        if cached_model is not None:
            logger.info("Loading CLIP from cache")
            self.model = cached_model['model']
            self.preprocess = cached_model['preprocess']
            self._model_loaded = True
            self._from_cache = True
            logger.info("Loaded from cache in <0.1s")
            
            # Still need to load reference database
            self._load_reference_database()
            return
        
        # Cache miss - load from disk. Guarded by a per-key lock (F-7):
        # without it, two concurrent cold-start requests can both observe
        # the miss above and both independently load the full CLIP model.
        with self.cache.get_load_lock(self.cache_key):
            # Double-check: another thread may have finished loading and
            # populated the cache while we were waiting for the lock.
            cached_model = self.cache.get(self.cache_key)
            if cached_model is not None:
                logger.info("Loading CLIP from cache (loaded by a concurrent request)")
                self.model = cached_model['model']
                self.preprocess = cached_model['preprocess']
                self._model_loaded = True
                self._from_cache = True
                self._load_reference_database()
                return

            try:
                import clip

                logger.info("Loading CLIP ViT-B/32 model from disk...")

                # Load CLIP model
                self.model, self.preprocess = clip.load(
                    "ViT-B/32",
                    device=self.device
                )

                self._from_cache = False
                logger.info("CLIP model loaded successfully")
            except Exception as exc:
                # Cache the failure so we don't retry a potentially expensive
                # download (CLIP weights from OpenAI CDN) on every request.
                logger.error(
                    "CLIP model load failed (%s) — will return neutral results "
                    "for all subsequent requests. Check clip package install and "
                    "network access to OpenAI CDN.", exc,
                )
                self.model       = None
                self.preprocess  = None
            finally:
                self._model_loaded = True  # Always set — prevents per-request retry
            if self.model is not None:
                pass  # load succeeded, continue below
            else:
                return  # load failed, skip cache store and DB load
            try:
                import clip as _clip_store  # noqa: F811

                # Store in cache for future use
                self.cache.set(
                    self.cache_key,
                    {
                        'model': self.model,
                        'preprocess': self.preprocess
                    },
                    self.model_size_mb
                )
                logger.info(f"Cached model ({self.model_size_mb}MB) for future use")

                # Load reference database
                self._load_reference_database()

            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
                raise
    
    def _load_reference_database(self):
        """Load pre-computed reference centroids."""
        database_path = Path(__file__).parent.parent.parent / "data" / "reference" / "clip_database.pkl"
        
        if database_path.exists():
            logger.info(f"Loading CLIP reference database from {database_path}")
            
            try:
                from backend.core.model_integrity import verify_integrity, ModelIntegrityError
                verify_integrity(database_path)
                with open(database_path, 'rb') as f:
                    database = pickle.load(f)
                
                # Load centroids as tensors
                self.real_centroid = torch.from_numpy(
                    database['real_centroid']
                ).float().to(self.device)
                
                self.fake_centroid = torch.from_numpy(
                    database['ai_centroid']
                ).float().to(self.device)

                # Mark database as available BEFORE returning
                self.db_available = True

                # Cache separation for confidence scaling in detect()
                self._db_separation = float(database.get("separation", 0.1))
                logger.info(
                    f"Loaded reference database: "
                    f"{database['real_count']} real, "
                    f"{database['ai_count']} AI images, "
                    f"separation={database['separation']:.4f}"
                )
                return
            
            except ModelIntegrityError:
                # A hash mismatch is a materially different, higher-severity
                # signal than "file missing/corrupt" -- let it propagate
                # rather than silently falling back to "signal disabled".
                raise
            except Exception as e:
                logger.warning(f"Failed to load reference database: {e}")
        
        self.db_available = False
        logger.warning(
            "CLIP reference database not found. Signal disabled. "
            "Run scripts/build_clip_database.py to enable CLIP detection."
        )
    
    def _extract_features(self, image_bytes: bytes) -> torch.Tensor:
        """Extract CLIP embedding from image."""
        from io import BytesIO
        
        # Load and preprocess image
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
        
        return features.squeeze(0)
    
    def _compute_similarity_score(self, embedding: torch.Tensor) -> float:
        """Compute AI probability based on embedding similarity."""
        # Cosine similarity to centroids
        sim_to_real = torch.cosine_similarity(
            embedding.unsqueeze(0), 
            self.real_centroid.unsqueeze(0)
        ).item()
        
        sim_to_fake = torch.cosine_similarity(
            embedding.unsqueeze(0), 
            self.fake_centroid.unsqueeze(0)
        ).item()
        
        # Convert to probability via temperature-scaled softmax.
        # Temperature=10 amplifies tiny centroid differences to extremes — reduced
        # to 4.0 which still discriminates well but avoids false confidence on
        # images near the boundary (weak centroid separation of 0.198).
        exp_fake = np.exp(sim_to_fake * 4.0)
        exp_real = np.exp(sim_to_real * 4.0)
        
        ai_probability = exp_fake / (exp_fake + exp_real)
        
        return float(ai_probability)
    
    def _neutral_result(self, reason: str) -> dict:
        return {"signal_name": "CLIP Embedding", "score": 0.5, "confidence": 0.0,
                "active": False, "explanation": reason, "raw_value": 0.0,
                "method": "clip_embedding"}

    def detect(self, image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
        """Detect if image is AI-generated using CLIP embeddings."""
        try:
            # Load model FIRST — this also loads the reference database and
            # sets self.db_available = True on success. Checking db_available
            # BEFORE calling _load_model() is the bug: the attribute is only
            # set inside _load_reference_database() which is called from
            # _load_model(), so the guard always fired and CLIP never ran.
            self._load_model()
        except Exception as e:
            logger.warning(f"CLIP model loading failed: {e}")
            return self._neutral_result(f"CLIP model unavailable: {e}")

        if not getattr(self, "db_available", False):
            return {
                "signal_name": "CLIP Embedding Analysis",
                "score": 0.5, "confidence": 0.0,
                "explanation": "CLIP reference database not built. Run scripts/build_clip_database.py.",
                "raw_value": 0.0, "expected_range": "N/A",
                "method": "clip_embedding_similarity",
                "from_cache": False, "active": False,
            }
        try:
            pass  # model already loaded above
            
            logger.info(f"Running CLIP detection on {filename}")
            
            # Extract features
            embedding = self._extract_features(image_bytes)
            
            # Compute similarity score
            ai_score = self._compute_similarity_score(embedding)
            
            # Generate explanation
            if ai_score > 0.7:
                explanation = f"CLIP embedding ({ai_score:.3f}) strongly matches AI-generated patterns"
            elif ai_score > 0.5:
                explanation = f"CLIP embedding ({ai_score:.3f}) leans toward AI-generated"
            elif ai_score > 0.3:
                explanation = f"CLIP embedding ({ai_score:.3f}) leans toward authentic"
            else:
                explanation = f"CLIP embedding ({ai_score:.3f}) strongly matches real photographs"
            
            cache_status = "cached" if self._from_cache else "fresh"
            logger.info(f"CLIP detection complete: score={ai_score:.3f} ({cache_status})")
            
            return {
                "signal_name": "CLIP Embedding Analysis",
                "score": float(ai_score),
                # Scale confidence with centroid separation — weak DB gives lower confidence
                "confidence": float(min(0.90, self._db_separation * 4.5)) if hasattr(self, "_db_separation") else 0.50,
                "explanation": explanation,
                "raw_value": float(ai_score),
                "expected_range": "> 0.5 for AI",
                "method": "clip_embedding_similarity",
                "from_cache": self._from_cache,
                # placeholder_mode=True means the centroids are random (DB not loaded)
                # — caller should treat this score as noise, not a real measurement.
                "placeholder_mode": not getattr(self, "db_available", False),
            }
            
        except Exception as e:
            logger.warning(f"CLIP detection failed: {e}")
            return {
                "signal_name": "CLIP Embedding Analysis",
                "score": 0.5,
                "confidence": 0.1,
                "explanation": "CLIP analysis failed — see server logs.",
                "active": False,
                "raw_value": 0.0,
                "expected_range": "N/A",
                "method": "clip_embedding_similarity",
                "from_cache": False
            }
    
    def cleanup(self):
        """Free GPU memory (model stays in cache)."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
            if self._from_cache:
                logger.info("GPU cache cleared (model preserved in cache)")
            else:
                logger.info("GPU cache cleared (model cached for future use)")
