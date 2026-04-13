"""
Production Readiness Checklist for VeriFile-X.

Usage:
    python scripts/production_check.py
    python scripts/production_check.py --strict
"""
import sys, json, logging, argparse, importlib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).parents[1]
PASSED, WARNED, FAILED = [], [], []

def check(name, cond, msg, warn=False):
    if cond:
        logger.info("PASS  %s", name); PASSED.append(name)
    elif warn:
        logger.warning("WARN  %s — %s", name, msg); WARNED.append((name, msg))
    else:
        logger.error("FAIL  %s — %s", name, msg); FAILED.append((name, msg))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    logger.info("VeriFile-X Production Readiness Check\n" + "=" * 50)

    logger.info("\n--- Reference Files ---")
    for fname, script in [
        ("data/reference/clip_database.pkl", "scripts/build_clip_database.py"),
        ("data/reference/ensemble_xgb.pkl", "scripts/train_ensemble.py"),
        ("data/reference/own_embedding_model.pt", "scripts/train_embedding.py"),
        ("data/reference/own_centroids.pkl", "scripts/build_centroids.py"),
    ]:
        p = ROOT / fname
        check(fname, p.exists() and p.stat().st_size > 100, f"Run {script}", warn=True)

    logger.info("\n--- Ensemble Sanity ---")
    rf = ROOT / "data" / "reference" / "ensemble_results.json"
    if rf.exists():
        r = json.loads(rf.read_text())
        check("cv_auc_not_perfect", r.get("cv_auc_mean", 0) < 0.99,
              f"CV AUC={r.get('cv_auc_mean',0):.4f} — likely data leakage")
        check("test_auc_present", "test_auc" in r, "Re-run train_ensemble.py", warn=True)
    else:
        check("ensemble_results.json", False, "Not found", warn=True)

    logger.info("\n--- Imports ---")
    sys.path.insert(0, str(ROOT))
    for svc in ["backend.main", "backend.core.config",
                 "backend.services.image_forensics",
                 "backend.utils.image_quality"]:
        try: importlib.import_module(svc); check(svc, True, "")
        except Exception as e: check(svc, False, str(e))

    logger.info("\n--- Config ---")
    try:
        from backend.core.config import settings
        check("VERSION_set", settings.VERSION != "0.0.0", "")
        check("CORS_set", len(settings.cors_origins_list) >= 1, "No CORS origins")
        check("DEBUG_false", not settings.DEBUG, "Set DEBUG=False", warn=True)
    except Exception as e:
        check("config", False, str(e))

    logger.info("\n--- Security ---")
    for fname in (".env", "api_keys.jsonl", "cases.jsonl"):
        check(f"{fname}_not_committed", not (ROOT / fname).exists(),
              f"{fname} should not be in repo", warn=True)

    logger.info("\n" + "=" * 50)
    logger.info("PASSED: %d | WARNED: %d | FAILED: %d", len(PASSED), len(WARNED), len(FAILED))
    if FAILED or (WARNED and args.strict): sys.exit(1)

if __name__ == "__main__":
    main()
