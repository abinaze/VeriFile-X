"""
OwnEmbeddingModel — lightweight EfficientNet-B0 fine-tuned for AI detection.

Replaces openai/CLIP (350MB, black box, requires internet).
EfficientNet-B0 is only 6MB, trains on CPU in under an hour,
and we own every weight.

Architecture:
  EfficientNet-B0 (pretrained on ImageNet)
  → remove final classifier
  → add 512-dim embedding layer
  → add binary classifier head (real vs AI)

Two modes:
  1. extract_embedding() — returns 512-dim vector (used like CLIP)
  2. classify() — returns probability directly
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# Path where trained weights are saved
MODEL_PATH = Path("data/reference/own_embedding_model.pt")

# Image preprocessing — same as EfficientNet expects
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class OwnEmbeddingModel(nn.Module):
    """
    EfficientNet-B0 with a 512-dim embedding head.
    Fine-tuned binary classifier: 0 = real, 1 = AI-generated.
    """

    def __init__(self, embedding_dim: int = 512, freeze_backbone: bool = False):
        super().__init__()

        # Load pretrained EfficientNet-B0
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Remove the final classifier (keeps feature extractor only)
        # EfficientNet-B0 outputs 1280-dim features before classifier
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        # Optional: freeze backbone and only train the head
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Embedding head: 1280 → 512
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classifier head: 512 → 1 (binary)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embedding: (batch, 512) — use for centroid comparison
            probability: (batch, 1) — direct AI probability
        """
        features  = self.features(x)
        pooled    = self.avgpool(features)
        embedding = self.embedding_head(pooled)
        prob      = self.classifier(embedding)
        return embedding, prob

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim embedding only (no gradient)."""
        with torch.no_grad():
            features  = self.features(x)
            pooled    = self.avgpool(features)
            embedding = self.embedding_head(pooled)
            # L2 normalize so cosine similarity works correctly
            embedding = nn.functional.normalize(embedding, dim=1)
        return embedding


def load_model(device: str = "cpu") -> OwnEmbeddingModel:
    """Load trained model from disk, or return untrained model if not found."""
    model = OwnEmbeddingModel()

    if MODEL_PATH.exists():
        logger.info(f"Loading trained model from {MODEL_PATH}")
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        logger.info("Model loaded successfully")
    else:
        logger.warning(
            f"No trained model found at {MODEL_PATH}. "
            "Run scripts/train_embedding.py first. "
            "Using untrained weights — results will be random."
        )

    model.eval()
    return model.to(device)
