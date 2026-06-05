"""
Build the CLIP centroid reference database from manifest.csv.

Reads real and AI images listed in the training split of manifest.csv,
computes CLIP ViT-B/32 embeddings, and saves balanced centroids to
data/reference/clip_database.pkl.

Usage:
    python scripts/build_clip_database.py
    python scripts/build_clip_database.py --max-per-class 5000
    python scripts/build_clip_database.py --model ViT-L/14 --max-per-class 3000
"""
import csv
import sys
import pickle
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[1]
MANIFEST  = ROOT / "data" / "manifest.csv"
OUT_PATH  = ROOT / "data" / "reference" / "clip_database.pkl"


# -- Helpers ----------------------------------------------------------------

def load_manifest(split: str = "train", max_per_class: int = 0,
                  seed: int = 42) -> tuple[list[Path], list[Path]]:
    """Return (real_paths, ai_paths) from manifest filtered to split."""
    if not MANIFEST.exists():
        logger.error(f"manifest.csv not found at {MANIFEST}")
        sys.exit(1)

    real_paths: list[Path] = []
    ai_paths:   list[Path] = []

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split", "train") != split:
                continue
            p = ROOT / Path(row["path"])
            if not p.exists():
                continue
            label = row.get("label", "")
            if label == "real":
                real_paths.append(p)
            elif label == "ai":
                ai_paths.append(p)

    rng = random.Random(seed)
    rng.shuffle(real_paths)
    rng.shuffle(ai_paths)

    if max_per_class > 0:
        real_paths = real_paths[:max_per_class]
        ai_paths   = ai_paths[:max_per_class]

    # Balance — use the same count for both classes
    n = min(len(real_paths), len(ai_paths))
    real_paths = real_paths[:n]
    ai_paths   = ai_paths[:n]

    logger.info(f"Loaded from manifest (split={split!r}): "
                f"{len(real_paths)} real, {len(ai_paths)} AI")
    return real_paths, ai_paths


def compute_embeddings(paths: list[Path], model, preprocess,
                       device: str, batch_size: int = 64) -> np.ndarray:
    """Compute normalised CLIP embeddings in batches."""
    all_embeddings: list[np.ndarray] = []
    failed = 0

    for i in tqdm(range(0, len(paths), batch_size),
                  desc="  Embedding batches", unit="batch"):
        batch_paths = paths[i : i + batch_size]
        tensors = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
            except Exception as exc:
                logger.debug(f"  Skip {p.name}: {exc}")
                failed += 1

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            emb = model.encode_image(batch_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)   # L2-normalise
        all_embeddings.append(emb.cpu().float().numpy())

    if failed:
        logger.warning(f"  Skipped {failed} images (unreadable)")

    if not all_embeddings:
        return np.empty((0,), dtype=np.float32)

    return np.vstack(all_embeddings)   # shape (N, D)


# -- Main -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CLIP centroid database from manifest.csv"
    )
    parser.add_argument(
        "--model", default="ViT-B/32",
        help="CLIP model variant. Default: ViT-B/32. Use ViT-L/14 for higher accuracy."
    )
    parser.add_argument(
        "--max-per-class", type=int, default=10000,
        help="Max images per class (real/AI). 0 = use all. Default: 10000."
    )
    parser.add_argument(
        "--split", default="train",
        help="Manifest split to use. Default: train."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="GPU batch size for embedding. Default: 64."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for balanced sampling. Default: 42."
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("VeriFile-X: CLIP Reference Database Builder")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}  max-per-class: {args.max_per_class}  "
                f"split: {args.split!r}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading CLIP {args.model} on {device}...")
    model, preprocess = clip.load(args.model, device=device)
    model.eval()
    logger.info("Model loaded.")

    # Load paths from manifest
    real_paths, ai_paths = load_manifest(
        split=args.split,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    if len(real_paths) == 0 or len(ai_paths) == 0:
        logger.error(
            "Not enough images found in manifest. "
            "Run the download scripts first and check that paths resolve on disk."
        )
        sys.exit(1)

    # Compute embeddings
    logger.info(f"Computing embeddings for {len(real_paths)} REAL images...")
    real_emb = compute_embeddings(real_paths, model, preprocess, device, args.batch_size)

    logger.info(f"Computing embeddings for {len(ai_paths)} AI images...")
    ai_emb = compute_embeddings(ai_paths, model, preprocess, device, args.batch_size)

    if real_emb.ndim < 2 or ai_emb.ndim < 2:
        logger.error("Embedding arrays are empty — no images were processed.")
        sys.exit(1)

    # Compute centroids
    logger.info("Computing centroids...")
    real_centroid = real_emb.mean(axis=0)                          # shape (D,)
    ai_centroid   = ai_emb.mean(axis=0)
    real_centroid = real_centroid / np.linalg.norm(real_centroid)  # L2-normalise
    ai_centroid   = ai_centroid   / np.linalg.norm(ai_centroid)

    # Cosine distance between centroids (higher = better separability)
    separation = float(1.0 - np.dot(real_centroid, ai_centroid))

    # Build database dict
    database = {
        "real_centroid":  real_centroid,
        "ai_centroid":    ai_centroid,
        "real_count":     int(len(real_emb)),
        "ai_count":       int(len(ai_emb)),
        "separation":     separation,
        "embedding_dim":  int(real_centroid.shape[0]),
        "model":          args.model,
    }

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(database, f)

    logger.info("=" * 70)
    logger.info("CLIP Database Built Successfully!")
    logger.info("=" * 70)
    logger.info(f"  Real images:       {database['real_count']}")
    logger.info(f"  AI images:         {database['ai_count']}")
    logger.info(f"  Embedding dim:     {database['embedding_dim']}")
    logger.info(f"  Centroid sep:      {database['separation']:.4f}  (>0.1 is good)")
    logger.info(f"  Saved to:          {OUT_PATH}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
