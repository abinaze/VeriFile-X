import sys
import csv
import pickle
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT        = Path(__file__).parents[1]
FEATURES    = ROOT / "data" / "features.csv"
MODEL_OUT   = ROOT / "data" / "reference" / "ensemble_xgb.pkl"
RESULTS_OUT = ROOT / "data" / "reference" / "ensemble_results.json"


def main():
    import xgboost as xgb
    import shap
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import roc_auc_score, f1_score
    import json

    logger.info("Loading feature matrix")
    rows, labels, feature_names = [], [], None

    with open(FEATURES, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if feature_names is None:
                feature_names = [k for k in row if k not in ("label", "path")]
            labels.append(int(row["label"]))
            rows.append([float(row[k]) for k in feature_names])

    X = np.array(rows)
    y = np.array(labels)
    logger.info(f"Feature matrix: {X.shape} | Positives: {y.sum()}/{len(y)}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    logger.info("Cross-validating (5-fold stratified)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=["roc_auc", "f1"],
        return_train_score=True,
    )

    auc  = scores["test_roc_auc"].mean()
    f1   = scores["test_f1"].mean()
    logger.info(f"CV AUC:  {auc:.4f} +/- {scores['test_roc_auc'].std():.4f}")
    logger.info(f"CV F1:   {f1:.4f} +/- {scores['test_f1'].std():.4f}")

    logger.info("Fitting final model on all data")
    model.fit(X, y)

    logger.info("Computing SHAP values")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_shap   = np.abs(shap_values).mean(axis=0)

    signal_importance = sorted(
        zip(feature_names, mean_shap.tolist()),
        key=lambda x: x[1], reverse=True
    )

    logger.info("Top 10 signals by SHAP importance:")
    for name, imp in signal_importance[:10]:
        logger.info(f"  {name:<45} {imp:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names, "explainer": explainer}, f)
    logger.info(f"Model saved to {MODEL_OUT}")

    results = {
        "cv_auc_mean":  round(auc, 4),
        "cv_auc_std":   round(scores["test_roc_auc"].std(), 4),
        "cv_f1_mean":   round(f1, 4),
        "cv_f1_std":    round(scores["test_f1"].std(), 4),
        "n_features":   len(feature_names),
        "n_samples":    len(y),
        "feature_importance": {k: round(v, 6) for k, v in signal_importance},
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {RESULTS_OUT}")
    logger.info("Phase 4 complete.")


if __name__ == "__main__":
    main()
