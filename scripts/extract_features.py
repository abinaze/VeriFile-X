import sys
import csv
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
OUT_CSV  = ROOT / "data" / "features.csv"
LIMIT    = 2000


def extract_one(image_path: Path, label: str) -> dict | None:
    try:
        from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
        img_bytes = image_path.read_bytes()
        det       = AdvancedEnsembleDetector(img_bytes, image_path.name)
        report    = det.detect()
        det.cleanup()

        row = {"label": 1 if label == "ai" else 0, "path": str(image_path)}
        for sig in report["all_signals"]:
            key       = sig["signal_name"].lower().replace(" ", "_")
            row[key]  = round(sig["score"], 6)
        return row
    except Exception as e:
        logger.warning(f"Failed {image_path.name}: {e}")
        return None


def main():
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Use train split for feature extraction (val/test reserved for evaluation)
        all_rows = [r for r in reader if r["split"] == "train"]

    import random
    random.seed(42)
    real_rows = [r for r in all_rows if r["label"] == "real"]
    ai_rows   = [r for r in all_rows if r["label"] == "ai"]
    random.shuffle(real_rows)
    random.shuffle(ai_rows)

    sample = real_rows[:LIMIT // 2] + ai_rows[:LIMIT // 2]
    random.shuffle(sample)

    logger.info(f"Extracting features from {len(sample)} images")

    for i, manifest_row in enumerate(sample):
        p = ROOT / manifest_row["path"].replace("\\", "/")
        if not p.exists():
            continue
        result = extract_one(p, manifest_row["label"])
        if result:
            rows.append(result)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(sample)} done, {len(rows)} successful")

    if not rows:
        logger.error("No features extracted")
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    logger.info(f"Saved {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
