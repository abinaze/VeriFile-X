"""
Fit Platt scaling calibration parameters from features.csv.

Usage:
    python scripts/fit_platt.py                     # fit on val split (default)
    python scripts/fit_platt.py --split val         # explicit val split
    python scripts/fit_platt.py --split train+val   # use all labelled rows
    python scripts/fit_platt.py --dry-run           # print params, do not save

Prerequisites:
    data/features.csv  — produced by scripts/extract_features.py
    data/manifest.csv  — provides split/label columns (optional fallback)

Output:
    data/reference/platt_params.json  — {"A": float, "B": float}
    Reload: restart the backend, or add a POST /api/v1/platt/reload endpoint.
"""
import csv
import json
import math
import logging
import argparse
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT       = Path(__file__).parents[1]
FEATURES   = ROOT / "data" / "features.csv"
MANIFEST   = ROOT / "data" / "manifest.csv"
PARAMS_OUT = ROOT / "data" / "reference" / "platt_params.json"
N_FEATURES = 30  # f0..f29


def _load_manifest_meta() -> dict:
    """Return {path: {split, label}} from manifest.csv if available."""
    meta = {}
    if not MANIFEST.exists():
        return meta
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        # Skip Git LFS pointer stubs
        first = f.read(50)
        if "git-lfs" in first:
            logger.warning("manifest.csv is a Git LFS stub — split info from features.csv only")
            return meta
        f.seek(0)
        for row in csv.DictReader(f):
            meta[row.get("path", "")] = {
                "split": row.get("split", ""),
                "label": row.get("label", ""),
            }
    return meta


def load_feature_matrix(split_filter: set) -> tuple:
    """
    Load feature rows for the requested splits.
    Returns (raw_scores, labels) as 1-D float64 numpy arrays.
    raw_score = mean(f0..f29) — a monotonic predictor suitable for Platt fit.
    """
    path_meta   = _load_manifest_meta()
    feat_cols   = [f"f{i}" for i in range(N_FEATURES)]
    scores, labels = [], []

    with open(FEATURES, newline="", encoding="utf-8") as f:
        first = f.read(50)
        if "git-lfs" in first:
            raise RuntimeError(
                "features.csv is a Git LFS pointer stub. "
                "Run: git lfs pull && python scripts/extract_features.py"
            )
        f.seek(0)
        for row in csv.DictReader(f):
            path  = row.get("path", "")
            meta  = path_meta.get(path, {})

            split = (meta.get("split")
                     or row.get("split", "")
                     or "train")
            if split_filter and split not in split_filter:
                continue

            label_str = (meta.get("label")
                         or row.get("label", "")
                         or "")
            if label_str in ("ai", "1"):
                label = 1
            elif label_str in ("real", "0"):
                label = 0
            else:
                continue  # unknown label — skip

            try:
                vals = [float(row.get(k, 0.5)) for k in feat_cols]
            except (ValueError, TypeError):
                continue

            scores.append(float(np.mean(vals)))
            labels.append(label)

    return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Fit Platt scaling parameters for VeriFile-X")
    parser.add_argument(
        "--split", default="val",
        help="Comma/plus-separated splits: train, val, test, train+val. Default: val",
    )
    parser.add_argument("--max-iter", type=int, default=500,
                        help="Gradient descent iterations. Default: 500")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate. Default: 0.01")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print params without writing platt_params.json")
    args = parser.parse_args()

    split_filter = {s.strip() for s in args.split.replace("+", ",").split(",")}
    logger.info("Loading features for splits: %s", split_filter)

    if not FEATURES.exists():
        raise FileNotFoundError(
            f"{FEATURES} not found. Run: python scripts/extract_features.py"
        )

    raw_scores, labels = load_feature_matrix(split_filter)

    if len(raw_scores) == 0:
        raise ValueError(
            f"No labelled rows found for splits {split_filter}. "
            "Check that manifest.csv has split/label columns and paths match "
            "features.csv, or pass --split train+val to use all labelled rows."
        )

    n_ai   = int(labels.sum())
    n_real = int((labels == 0).sum())
    logger.info("Loaded %d samples  (AI=%d  real=%d)  mean_score=%.4f",
                len(raw_scores), n_ai, n_real, float(raw_scores.mean()))

    if n_ai == 0 or n_real == 0:
        raise ValueError(
            "Both AI and real samples are required for Platt fitting. "
            f"Found AI={n_ai}, real={n_real}."
        )

    import sys
    sys.path.insert(0, str(ROOT))
    from backend.services.platt_calibrator import fit

    logger.info("Fitting Platt parameters (max_iter=%d, lr=%.4f)…", args.max_iter, args.lr)
    A, B = fit(raw_scores, labels, max_iter=args.max_iter, lr=args.lr)

    def _sig(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))

    p0  = _sig(A * 0.0 + B)
    p05 = _sig(A * 0.5 + B)
    p1  = _sig(A * 1.0 + B)
    logger.info("Fitted A=%.6f  B=%.6f", A, B)
    logger.info("Sanity: calibrate(0.0)=%.3f  calibrate(0.5)=%.3f  calibrate(1.0)=%.3f",
                p0, p05, p1)

    if p05 < 0.35 or p05 > 0.65:
        logger.warning(
            "calibrate(0.5) = %.3f is far from 0.5 — the val set may be "
            "imbalanced or raw scores may not be centred at 0.5. "
            "Consider using --split train+val for more data.", p05
        )

    if args.dry_run:
        logger.info("--dry-run active: not writing params")
        return

    PARAMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"A": round(float(A), 6), "B": round(float(B), 6)}
    PARAMS_OUT.write_text(json.dumps(payload, indent=2))
    logger.info("Written: %s", PARAMS_OUT)
    logger.info("Restart the backend (or add a /api/v1/platt/reload endpoint) to apply.")


if __name__ == "__main__":
    main()
