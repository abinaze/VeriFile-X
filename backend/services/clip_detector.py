"""
CLIP-based Universal Fake Detection with proper reference database.
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

logger = setup_logger(__name__)


class CLIPDetector:
    """CLIP-based universal AI detection with learned centroids."""
    
    def __init__(self):
        """Initialize CLIP detector."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self._model_loaded = False
        
        # Reference centroids (will be loaded from database)
        self.real_centroid = None
        self.fake_centroid = None
        
        logger.info(f"CLIP Detector initialized (device: {self.device})")
    
    def _load_model(self):
        """Lazy load CLIP model and reference database."""
        if self._model_loaded:
            return
        
        try:
            import clip
            
            logger.info("Loading CLIP ViT-B/32 model...")
            
            # Load CLIP model
            self.model, self.preprocess = clip.load(
                "ViT-B/32", 
                device=self.device
            )
            
            self._model_loaded = True
            logger.info("CLIP model loaded successfully")
            
            # Load reference database
            self._load_reference_database()
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _load_reference_database(self):
        """Load pre-computed reference centroids."""
        database_path = Path("data/reference/clip_database.pkl")
        
        if database_path.exists():
            logger.info(f"Loading CLIP reference database from {database_path}")
            
            try:
                with open(database_path, 'rb') as f:
                    database = pickle.load(f)
                
                # Load centroids as tensors
                self.real_centroid = torch.from_numpy(
                    database['real_centroid']
                ).float().to(self.device)
                
                self.fake_centroid = torch.from_numpy(
                    database['ai_centroid']
                ).float().to(self.device)
                
                logger.info(
                    f"Loaded reference database: "
                    f"{database['real_count']} real, "
                    f"{database['ai_count']} AI images, "
                    f"separation={database['separation']:.4f}"
                )
                return
            
            except Exception as e:
                logger.warning(f"Failed to load reference database: {e}")
        
        # Fallback to placeholder values
        logger.warning(
            "Reference database not found, using placeholder centroids. "
            "Run 'python scripts/build_clip_database.py' for better accuracy."
        )
        self._initialize_placeholder_centroids()
    
    def _initialize_placeholder_centroids(self):
        """Initialize placeholder centroids (fallback)."""
        embedding_dim = 512  # ViT-B/32 embedding size
        
        # Random initialization (will be replaced by actual data)
        self.real_centroid = torch.randn(embedding_dim).to(self.device) * 0.01
        self.fake_centroid = torch.randn(embedding_dim).to(self.device) * 0.01
        
        # Ensure they're different
        self.fake_centroid += torch.ones(embedding_dim).to(self.device) * 0.1
        
        # Normalize
        self.real_centroid = self.real_centroid / self.real_centroid.norm()
        self.fake_centroid = self.fake_centroid / self.fake_centroid.norm()
        
        logger.info("Initialized placeholder centroids (run build_clip_database.py for production)")
    
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
        
        # Convert to probability via softmax-like formula
        # Higher similarity to fake centroid = higher AI probability
        exp_fake = np.exp(sim_to_fake * 10)  # Temperature scaling
        exp_real = np.exp(sim_to_real * 10)
        
        ai_probability = exp_fake / (exp_fake + exp_real)
        
        return float(ai_probability)
    
    def detect(self, image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
        """Detect if image is AI-generated using CLIP embeddings."""
        try:
            # Lazy load model
            self._load_model()
            
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
            
            logger.info(f"CLIP detection complete: score={ai_score:.3f}")
            
            return {
                "signal_name": "CLIP Embedding Analysis",
                "score": float(ai_score),
                "confidence": 0.90,  # High confidence with real database
                "explanation": explanation,
                "raw_value": float(ai_score),
                "expected_range": "> 0.5 for AI",
                "method": "clip_embedding_similarity"
            }
            
        except Exception as e:
            logger.warning(f"CLIP detection failed: {e}")
            return {
                "signal_name": "CLIP Embedding Analysis",
                "score": 0.5,  # Neutral score on failure
                "confidence": 0.1,
                "explanation": f"Analysis failed: {str(e)}",
                "raw_value": 0.0,
                "expected_range": "N/A",
                "method": "clip_embedding_similarity"
            }
    
    def cleanup(self):
        """Free GPU memory."""
        if self._model_loaded and self.device == "cuda":
            del self.model
            torch.cuda.empty_cache()
            self._model_loaded = False
            logger.info("CLIP model unloaded")
