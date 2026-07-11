import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent.parent / "data" / "reference" / "own_embedding_model.pt"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class OwnEmbeddingModel(nn.Module):

    def __init__(self, embedding_dim: int = 512, freeze_backbone: bool = False,
                 skip_pretrained: bool = False):
        """
        Args:
            skip_pretrained: if True, skip the ImageNet-pretrained weight
                download entirely. BUG FIX: load_model() (below) always
                constructs this model and then IMMEDIATELY overwrites every
                one of its weights with the real trained checkpoint from
                own_embedding_model.pt — meaning the pretrained-weight
                download was previously performed and then 100% discarded
                on every single load, wasting a network round-trip (only
                mitigated by torch hub's local disk cache, which won't
                survive an ephemeral/container-restart deployment).
                load_model() now passes skip_pretrained=True.
        """
        super().__init__()

        backbone = models.efficientnet_b0(weights=None)
        if not skip_pretrained:
            try:
                import torch.hub as hub
                state = hub.load_state_dict_from_url(
                    "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
                    check_hash=False,
                )
                backbone.load_state_dict(state)
            except Exception as e:
                logger.warning(f"Could not load pretrained weights: {e}. Using random init.")

        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features  = self.features(x)
        pooled    = self.avgpool(features)
        embedding = self.embedding_head(pooled)
        prob      = self.classifier(embedding)
        return embedding, prob

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features  = self.features(x)
            pooled    = self.avgpool(features)
            embedding = self.embedding_head(pooled)
            embedding = nn.functional.normalize(embedding, dim=1)
        return embedding


def load_model(device: str = "cpu") -> Optional[OwnEmbeddingModel]:
    if not MODEL_PATH.exists():
        logger.warning(
            f"No trained model found at {MODEL_PATH}. "
            "Run scripts/train_embedding.py first. "
            "Returning None — own_embedding signal will return neutral 0.5."
        )
        return None

    # skip_pretrained=True: the real trained checkpoint loaded two lines
    # below overwrites every weight anyway — no point downloading ImageNet
    # pretrained weights first.
    model = OwnEmbeddingModel(skip_pretrained=True)
    logger.info(f"Loading trained model from {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Model loaded successfully")
    return model.to(device)
