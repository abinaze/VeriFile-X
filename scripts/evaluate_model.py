"""
Model Evaluation and Benchmarking Script.

Implements diagnostics from the ML accuracy report:
- Confusion matrix, precision, recall, F1, AUROC
- Class balance analysis
- Data leakage check (train/val/test path overlap)
- RandomizedSearchCV hyperparameter search

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --hparam-search
"""
import csv
import json
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

ROOT        = Path(__file__).parents[1]
MANIFEST    = ROOT / "data" / "manifest.csv"
FEATURES    = ROOT / "data" / "features.csv"
RESULTS_OUT = ROOT / "data" / "reference" / "eval_results.json"


def check_class_balance(manifest_path: Path) -> dict:
    counts = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("label", "")
            counts[label] = counts.get(label, 0) + 1

    total         = sum(counts.values())
    balance_ratio = min(counts.values()) / max(counts.values(), 1)
    imbalanced    = balance_ratio < 0.30

    logger.info(f"Class balance: {counts} (ratio={balance_ratio:.3f})")
    if imbalanced:
        logger.warning(
            f"Class imbalance detected (ratio={balance_ratio:.3f}). "
            "Consider resampling or scale_pos_weight."
        )
    return {"counts": counts, "total": total,
            "balance_ratio": round(balance_ratio, 4), "imbalanced": imbalanced}


def check_data_leakage(manifest_path: Path) -> dict:
    split_paths: dict = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "train")
            path  = row.get("path", "")
            if split not in split_paths:
                split_paths[split] = set()
            split_paths[split].add(path)

    overlaps = {}
    splits   = list(split_paths.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b    = splits[i], splits[j]
            overlap = split_paths[a] & split_paths[b]
            if overlap:
                key           = f"{a}_vs_{b}"
                overlaps[key] = len(overlap)
                logger.warning(f"Data leakage: {len(overlap)} duplicate paths between {a} and {b}")

    leakage = len(overlaps) > 0
    if not leakage:
        logger.info("No path-based data leakage detected.")
    return {"split_sizes": {k: len(v) for k, v in split_paths.items()},
            "leakage_detected": leakage, "overlapping_paths": overlaps}


def evaluate_xgboost(features_path: Path, threshold: float = 0.5) -> dict:
    import pickle
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, classification_report,
    )

    model_path = ROOT / "data" / "reference" / "ensemble_xgb.pkl"
    if not model_path.exists():
        logger.warning("ensemble_xgb.pkl not found. Run scripts/train_ensemble.py first.")
        return {"error": "Model not found"}

    with open(model_path, "rb") as f:
        pkg = pickle.load(f)
    model         = pkg["model"]
    feature_names = pkg["feature_names"]

    rows, labels = [], []
    with open(features_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
            rows.append([float(row.get(k, 0.5)) for k in feature_names])

    X = np.array(rows)
    y = np.array(labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, test_idx    = next(sss.split(X, y))
    X_test, y_test = X[test_idx], y[test_idx]

    y_pred  = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    y_score = model.predict_proba(X_test)[:, 1]

    acc   = float(accuracy_score(y_test, y_pred))
    prec  = float(precision_score(y_test, y_pred, zero_division=0))
    rec   = float(recall_score(y_test, y_pred, zero_division=0))
    f1    = float(f1_score(y_test, y_pred, zero_division=0))
    auroc = float(roc_auc_score(y_test, y_score))
    cm    = confusion_matrix(y_test, y_pred).tolist()

    alarms = []
    if acc   < 0.90: alarms.append(f"Accuracy {acc:.3f} below threshold 0.90")
    if prec  < 0.85: alarms.append(f"Precision {prec:.3f} below threshold 0.85")
    if rec   < 0.80: alarms.append(f"Recall {rec:.3f} below threshold 0.80")
    if f1    < 0.83: alarms.append(f"F1 {f1:.3f} below threshold 0.83")
    if auroc < 0.92: alarms.append(f"AUROC {auroc:.3f} below threshold 0.92")

    logger.info(f"Evaluation on {len(y_test)} held-out samples:")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  F1:        {f1:.4f}")
    logger.info(f"  AUROC:     {auroc:.4f}")
    for alarm in alarms:
        logger.warning(f"ALARM: {alarm}")
    if not alarms:
        logger.info("All metrics above alarm thresholds.")

    return {
        "n_test_samples": len(y_test), "threshold": threshold,
        "accuracy": round(acc, 4), "precision_ai": round(prec, 4),
        "recall_ai": round(rec, 4), "f1_ai": round(f1, 4),
        "auroc": round(auroc, 4), "confusion_matrix": cm,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["real", "ai"], output_dict=True),
        "alarms": alarms,
        "targets":    {"accuracy": 0.95, "precision_ai": 0.90,
                       "recall_ai": 0.88, "f1_ai": 0.89, "auroc": 0.97},
        "thresholds": {"accuracy": 0.90, "precision_ai": 0.85,
                       "recall_ai": 0.80, "f1_ai": 0.83, "auroc": 0.92},
    }


def hyperparameter_search(features_path: Path, n_iter: int = 20) -> dict:
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    rows, labels, feature_names = [], [], None
    with open(features_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if feature_names is None:
                feature_names = [k for k in row if k not in ("label", "path")]
            labels.append(int(row["label"]))
            rows.append([float(row[k]) for k in feature_names])

    X = np.array(rows)
    y = np.array(labels)

    param_grid = {
        "learning_rate":    [0.01, 0.05, 0.10, 0.15, 0.20],
        "max_depth":        [3, 4, 5, 6],
        "n_estimators":     [100, 200, 300, 400],
        "subsample":        [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma":            [0, 0.1, 0.2, 0.5],
    }

    model  = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logger.info(f"Running RandomizedSearchCV with {n_iter} iterations")
    search = RandomizedSearchCV(model, param_grid, n_iter=n_iter,
                                cv=cv, scoring="roc_auc",
                                random_state=42, n_jobs=-1, verbose=1)
    search.fit(X, y)
    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV AUC: {search.best_score_:.4f}")
    return {"best_params": search.best_params_,
            "best_cv_auc": round(search.best_score_, 4), "n_iterations": n_iter}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--hparam-search", action="store_true")
    parser.add_argument("--n-iter",       type=int, default=20)
    args = parser.parse_args()

    results = {}

    if MANIFEST.exists():
        logger.info("=== CLASS BALANCE CHECK ===")
        results["class_balance"] = check_class_balance(MANIFEST)
        logger.info("=== DATA LEAKAGE CHECK ===")
        results["leakage_check"] = check_data_leakage(MANIFEST)
    else:
        logger.warning("manifest.csv not found")

    if FEATURES.exists():
        logger.info("=== MODEL EVALUATION ===")
        results["evaluation"] = evaluate_xgboost(FEATURES, threshold=args.threshold)
        if args.hparam_search:
            logger.info("=== HYPERPARAMETER SEARCH ===")
            results["hparam_search"] = hyperparameter_search(FEATURES, n_iter=args.n_iter)
    else:
        logger.warning("features.csv not found — run scripts/extract_features.py first")

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {RESULTS_OUT}")


if __name__ == "__main__":
    main()
