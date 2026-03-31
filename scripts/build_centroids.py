import sys
import csv
import pickle
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
from backend.services.own_detector.model import load_model, TRANSFORM

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT         = Path(__file__).parents[1]
MANIFEST     = ROOT / "data" / "manifest.csv"
OUTPUT_PATH  = ROOT / "data" / "reference" / "own_centroids.pkl"
SAMPLES_EACH = 5000


def extract_embeddings(rows, device, model):
    embeddings = []
    for row in tqdm(rows, desc="Extracting embeddings"):
        img_path = ROOT / row["path"].replace("\\", "/")
        try:
            img    = Image.open(img_path).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0).to(device)
            emb    = model.extract_embedding(tensor)
            embeddings.append(emb.cpu().numpy().squeeze())
        except Exception:
            continue
    return np.array(embeddings)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info("Loading trained model")
    model = load_model(device)
    model.eval()

    logger.info("Reading manifest")
    real_rows, ai_rows = [], []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = ROOT / row["path"].replace("\\", "/")
            if not p.exists():
                continue
            if row["label"] == "real" and row["split"] == "train":
                real_rows.append(row)
            elif row["label"] == "ai" and row["split"] == "train":
                ai_rows.append(row)

    import random
    random.seed(42)
    random.shuffle(real_rows)
    random.shuffle(ai_rows)
    real_sample = real_rows[:SAMPLES_EACH]
    ai_sample   = ai_rows[:SAMPLES_EACH]

    logger.info(f"Extracting real embeddings ({len(real_sample)} images)")
    real_embeddings = extract_embeddings(real_sample, device, model)
    logger.info(f"Extracting AI embeddings ({len(ai_sample)} images)")
    ai_embeddings   = extract_embeddings(ai_sample, device, model)

    real_centroid = real_embeddings.mean(axis=0)
    ai_centroid   = ai_embeddings.mean(axis=0)

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    separation = cosine_sim(real_centroid, ai_centroid)
    logger.info(f"Centroid cosine similarity (lower = better separation): {separation:.4f}")

    database = {
        "real_centroid": real_centroid.astype(np.float32),
        "ai_centroid":   ai_centroid.astype(np.float32),
        "real_count":    len(real_embeddings),
        "ai_count":      len(ai_embeddings),
        "separation":    separation,
        "model":         "own_embedding_efficientnet_b0",
        "embedding_dim": real_centroid.shape[0],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(database, f)

    logger.info(f"Saved centroids to {OUTPUT_PATH}")
    logger.info(f"Real centroid shape: {real_centroid.shape}")
    logger.info(f"Centroid separation: {separation:.4f}")


if __name__ == "__main__":
    main()
