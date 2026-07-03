"""
Train XGBoost ensemble classifier with ML accuracy improvements:
- scale_pos_weight for class imbalance
- Early stopping on validation AUC
- Regularization: gamma, min_child_weight
- Separate held-out test evaluation
"""
import csv
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT        = Path(__file__).parents[1]
FEATURES    = ROOT / "data" / "features.csv"
MODEL_OUT   = ROOT / "data" / "reference" / "ensemble_xgb.pkl"
RESULTS_OUT = ROOT / "data" / "reference" / "ensemble_results.json"


def main():
    import xgboost as xgb
    import shap
    from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate, train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size",  type=float, default=0.15)
    parser.add_argument("--early-stop", type=int,   default=20)
    args = parser.parse_args()

    logger.info("Loading feature matrix")
    rows, labels, sources, feature_names = [], [], [], None

    with open(FEATURES, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if feature_names is None:
                feature_names = [k for k in row if k not in ("label", "path", "source")]
            labels.append(int(row["label"]))
            rows.append([float(row[k]) for k in feature_names])
            sources.append(row.get("source", "unknown"))

    X = np.array(rows)
    y = np.array(labels)
    logger.info(f"Feature matrix: {X.shape} | Positives: {y.sum()}/{len(y)}")

    neg_count        = (y == 0).sum()
    pos_count        = y.sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    logger.info(f"scale_pos_weight: {scale_pos_weight:.3f}")

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    cv_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42,
    )

    logger.info("Cross-validating (5-fold stratified)")
    sources_arr = np.array(sources)
    unique_sources = np.unique(sources_arr)
    if len(unique_sources) > 1:
        cv = GroupKFold(n_splits=5)
        cv_groups = sources_arr
        cv_method = "GroupKFold"
        logger.info(
            "Using GroupKFold by source (%d unique sources: %s) to prevent "
            "generator-family leakage from inflating CV scores.",
            len(unique_sources), list(unique_sources)[:10],
        )
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_groups = None
        cv_method = "StratifiedKFold"
        logger.info("Only one source present — falling back to StratifiedKFold.")
    scores = cross_validate(cv_model, X_dev, y_dev, cv=cv, groups=cv_groups,
                            scoring=["roc_auc", "f1"], return_train_score=True)
    auc_cv = scores["test_roc_auc"].mean()
    f1_cv  = scores["test_f1"].mean()
    logger.info(f"CV AUC: {auc_cv:.4f} +/- {scores['test_roc_auc'].std():.4f}")
    logger.info(f"CV F1:  {f1_cv:.4f}  +/- {scores['test_f1'].std():.4f}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15, stratify=y_dev, random_state=0
    )

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=args.early_stop,
        random_state=42,
    )
    logger.info("Fitting final model with early stopping")
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
    logger.info(f"Best iteration: {model.best_iteration}")

    y_score   = model.predict_proba(X_test)[:, 1]
    y_pred    = (y_score >= 0.5).astype(int)
    test_auc  = float(roc_auc_score(y_test, y_score))
    test_f1   = float(f1_score(y_test, y_pred, zero_division=0))
    test_acc  = float(accuracy_score(y_test, y_pred))
    logger.info(f"Test AUC: {test_auc:.4f}  F1: {test_f1:.4f}  Acc: {test_acc:.4f}")

    logger.info("Computing SHAP values")
    try:
        # shap.Explainer handles XGBoost 2.x base_score format correctly
        explainer   = shap.Explainer(model)
        shap_values = explainer(X_dev).values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]   # binary: take class-1 slice
        mean_shap        = np.abs(shap_values).mean(axis=0)
        signal_importance = sorted(zip(feature_names, mean_shap.tolist()),
                                   key=lambda x: x[1], reverse=True)
        logger.info("Top 10 signals by SHAP importance:")
        for name, imp in signal_importance[:10]:
            logger.info(f"  {name:<45} {imp:.4f}")
        shap_ok = True
    except Exception as exc:
        logger.warning(f"SHAP failed ({exc}) — falling back to XGBoost native importance")
        native_imp = model.get_booster().get_score(importance_type="gain")
        signal_importance = sorted(native_imp.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top 10 signals by gain importance:")
        for name, imp in signal_importance[:10]:
            logger.info(f"  {name:<45} {imp:.4f}")
        explainer = None
        shap_ok   = False

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names,
                     "explainer": explainer}, f)
    logger.info(f"Model saved to {MODEL_OUT}")

    results = {
        "cv_method":         cv_method,
        "cv_auc_mean":       round(auc_cv, 4),
        "cv_auc_std":        round(scores["test_roc_auc"].std(), 4),
        "cv_f1_mean":        round(f1_cv, 4),
        "cv_f1_std":         round(scores["test_f1"].std(), 4),
        "test_auc":          round(test_auc, 4),
        "test_f1":           round(test_f1, 4),
        "test_accuracy":     round(test_acc, 4),
        "best_iteration":    int(model.best_iteration),
        "scale_pos_weight":  round(float(scale_pos_weight), 4),
        "n_features":        len(feature_names),
        "n_samples":         len(y),
        "feature_names":     feature_names,
        "feature_importance": {k: round(v, 6) for k, v in signal_importance},
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    if results["cv_auc_mean"] > 0.995:
        logger.warning(
            "CV AUC = 1.0 — likely data leakage. "
            "CIFAKE images are 32x32 while COCO images are large JPEGs. "
            "The model may be learning resolution/compression, not AI signals. "
            "Consider adding more diverse datasets (ArtiFact, Defactify) with "
            "matched resolutions before trusting these results in production."
        )
    logger.info(f"Results saved to {RESULTS_OUT}")


if __name__ == "__main__":
    main()
