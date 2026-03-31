"""
Train OwnEmbeddingModel on manifest.csv dataset.

Usage:
    python scripts/train_embedding.py
    python scripts/train_embedding.py --epochs 10 --batch 32 --limit 50000

What this does:
    1. Reads manifest.csv
    2. Loads images in batches
    3. Trains EfficientNet-B0 binary classifier (real=0, AI=1)
    4. Saves best model to data/reference/own_embedding_model.pt
    5. Logs accuracy/loss to tensorboard (optional)
"""
import sys
import csv
import time
import random
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from backend.services.own_detector.model import OwnEmbeddingModel, MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT          = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "data" / "manifest.csv"

# ── Image transform ────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)
    ], p=0.5),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([transforms.RandomAffine(degrees=10, translate=(0.05, 0.05))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Dataset ────────────────────────────────────────────────────────────────

class ImageManifestDataset(Dataset):
    """Reads images listed in manifest.csv."""

    def __init__(self, rows: List[dict], transform, root: Path):
        self.rows      = rows
        self.transform = transform
        self.root      = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row   = self.rows[idx]
        label = 1 if row["label"] == "ai" else 0

        # Fix Windows backslash paths
        img_path = self.root / row["path"].replace("\\", "/")

        try:
            img = Image.open(img_path).convert("RGB")
            return self.transform(img), torch.tensor(label, dtype=torch.float32)
        except Exception:
            # Return black image on failure — rare corrupt file
            blank = torch.zeros(3, 224, 224)
            return blank, torch.tensor(label, dtype=torch.float32)


# ── Load manifest ──────────────────────────────────────────────────────────

def load_manifest(limit: int = 0) -> Tuple[List[dict], List[dict]]:
    if not MANIFEST_PATH.exists():
        logger.error(f"manifest.csv not found at {MANIFEST_PATH}")
        sys.exit(1)

    all_rows = []
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_path = ROOT / row["path"].replace("\\", "/")
            if not img_path.exists():
                continue
            w = int(row.get("width", 0))
            h = int(row.get("height", 0))
            if w < 64 or h < 64:
                continue
            all_rows.append(row)

    real_all = [r for r in all_rows if r["label"] == "real"]
    ai_all   = [r for r in all_rows if r["label"] == "ai"]

    logger.info(f"All on disk — real: {len(real_all)}, AI: {len(ai_all)}")

    random.seed(42)
    random.shuffle(real_all)
    random.shuffle(ai_all)

    val_size = min(5000, len(real_all) // 10, len(ai_all) // 10)
    real_val, real_train = real_all[:val_size], real_all[val_size:]
    ai_val,   ai_train   = ai_all[:val_size],   ai_all[val_size:]

    val_rows = real_val + ai_val
    random.shuffle(val_rows)

    min_count = min(len(real_train), len(ai_train))
    if limit > 0:
        min_count = min(min_count, limit // 2)

    balanced = real_train[:min_count] + ai_train[:min_count]
    random.shuffle(balanced)

    logger.info(f"Train: {len(balanced)} ({min_count} real + {min_count} AI)")
    logger.info(f"Val:   {len(val_rows)} ({val_size} real + {val_size} AI)")

    return balanced, val_rows

# ── Training loop ──────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load data
    train_rows, val_rows = load_manifest(limit=args.limit)

    if len(train_rows) == 0:
        logger.error(
            "No training images found on disk. "
            "Run download scripts first to get images locally."
        )
        sys.exit(1)

    train_ds = ImageManifestDataset(train_rows, TRAIN_TRANSFORM, ROOT)
    val_ds   = ImageManifestDataset(val_rows,   VAL_TRANSFORM,   ROOT)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch,
        shuffle=False, num_workers=0,
    )

    logger.info(f"Train batches: {len(train_loader)} "
                f"| Val batches: {len(val_loader)}")

    # Model
    model = OwnEmbeddingModel(freeze_backbone=False)
    model = model.to(device)

    # Loss and optimiser
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=1e-3
)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_acc  = 0.0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            _, probs = model(images)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            preds          = (probs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Epoch {epoch} batch {batch_idx+1}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        train_acc  = train_correct / train_total * 100
        train_loss = train_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                _, probs = model(images)
                loss = criterion(probs, labels)
                val_loss    += loss.item()
                preds        = (probs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc  = val_correct / val_total * 100 if val_total > 0 else 0
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{args.epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.1f}% "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.1f}% "
            f"| {elapsed:.0f}s"
        )

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info(f"  ✅ Best model saved (val_acc={val_acc:.1f}%)")

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.1f}%")
    logger.info(f"Model saved to: {MODEL_PATH}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OwnEmbeddingModel")
    parser.add_argument("--epochs",          type=int,   default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch",           type=int,   default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr",              type=float, default=3e-4,
                        help="Learning rate (default: 0.0001)")
    parser.add_argument("--limit",           type=int,   default=0,
                        help="Limit images per class, 0=use all")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze EfficientNet backbone, train head only")
    args = parser.parse_args()

    logger.info("=== VeriFile-X Embedding Detector Training ===")
    logger.info(f"Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    logger.info(f"Limit: {args.limit if args.limit else 'all'}")

    train(args)
