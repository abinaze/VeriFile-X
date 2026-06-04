"""
Master orchestration script — runs the full VeriFile-X training pipeline
in the correct order on a single machine.

Usage:
    python scripts/retrain_all.py                         # full pipeline
    python scripts/retrain_all.py --dry-run               # print steps, do nothing
    python scripts/retrain_all.py --start-at step3        # resume after a crash
    python scripts/retrain_all.py --skip step1,step2      # skip download steps

Steps:
    step1  — Download AI datasets   (CIFAKE + optionally GAN/DiffusionDB)
    step2  — Download real datasets (COCO real + optionally DIV2K / RAISE1K)
    step3  — Train EfficientNet-B0 embedding detector
    step4  — Build CLIP centroid database
    step5  — Build OwnEmbedding centroid database
    step6  — Extract 30-signal feature vectors
    step7  — Train XGBoost meta-model

RTX 4050 (6 GB VRAM) recommended settings are used automatically unless
overridden with the flags documented below.

After completion:
    • Copy data/reference/*.pkl and data/reference/*.pt to the running server
    • Restart the backend and check GET /health — all models should be "ok"
    • Run: pytest backend/tests/ -v -m "not slow"
"""
import os
import sys

# Force UTF-8 output on Windows (cp1252 terminal crashes on box-drawing chars)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import time
import shutil
import logging
import argparse
import subprocess
import platform
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# -- Logging to both console and file --------------------------------------
ROOT     = Path(__file__).parents[1]
LOG_FILE = ROOT / "data" / "training_log.txt"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

SCRIPTS     = ROOT / "scripts"
DATA        = ROOT / "data"
MANIFEST    = DATA / "manifest.csv"
FEATURES    = DATA / "features.csv"
REFERENCE   = DATA / "reference"

# -- Expected output files for each step (used to detect completion) -------
STEP_OUTPUTS: dict[str, list[Path]] = {
    "step1": [DATA / "ai" / "cifake"],
    "step2": [DATA / "real" / "coco_val"],
    "step3": [REFERENCE / "own_embedding_model.pt"],
    "step4": [REFERENCE / "clip_database.pkl"],
    "step5": [REFERENCE / "own_centroids.pkl"],
    "step6": [FEATURES],
    "step7": [REFERENCE / "ensemble_xgb.pkl",
               REFERENCE / "ensemble_results.json"],
}

STEP_NAMES = {
    "step1": "Download AI datasets (CIFAKE)",
    "step2": "Download real datasets (COCO val2017)",
    "step3": "Train EfficientNet-B0 embedding detector",
    "step4": "Build CLIP centroid database",
    "step5": "Build OwnEmbedding centroid database",
    "step6": "Extract 30-signal feature vectors",
    "step7": "Train XGBoost meta-model",
}

# Rough time estimates per step (minutes) — for planning / ETA display
STEP_ETA_MIN = {
    "step1": 30,
    "step2": 20,
    "step3": 90,
    "step4": 40,
    "step5": 20,
    "step6": 180,   # depends heavily on --limit
    "step7": 15,
}


# -- Utilities --------------------------------------------------------------

def _python() -> str:
    """Return the Python executable to use (same as the current interpreter)."""
    return sys.executable


def _check_gpu() -> bool:
    """Return True if a CUDA-capable GPU is visible to PyTorch."""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            logger.info(f"GPU: {name}  VRAM: {vram} GB")
        else:
            logger.warning("No CUDA GPU detected — training will be slow on CPU.")
        return has_gpu
    except ImportError:
        logger.warning("PyTorch not importable — cannot check GPU.")
        return False


def _check_disk(min_gb: float = 10.0) -> float:
    """Return free disk space in GB and warn if below min_gb."""
    try:
        usage  = shutil.disk_usage(ROOT)
        free   = usage.free / (1024 ** 3)
        total  = usage.total / (1024 ** 3)
        logger.info(f"Disk: {free:.1f} GB free of {total:.1f} GB total")
        if free < min_gb:
            logger.warning(
                f"Only {free:.1f} GB free — recommend at least {min_gb:.0f} GB. "
                "Some steps may fail."
            )
        return free
    except Exception as exc:
        logger.warning(f"Could not check disk space: {exc}")
        return 999.0


def _count_manifest_rows() -> tuple[int, int]:
    """Return (real_count, ai_count) from manifest.csv."""
    if not MANIFEST.exists():
        return 0, 0
    real, ai = 0, 0
    try:
        import csv
        with open(MANIFEST, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("label") == "real":
                    real += 1
                elif row.get("label") == "ai":
                    ai += 1
    except Exception:
        pass
    return real, ai


def _step_already_done(step: str) -> bool:
    """
    Return True if all expected outputs for *step* exist (non-empty).
    This allows --start-at to skip already-completed steps.
    """
    outputs = STEP_OUTPUTS.get(step, [])
    if not outputs:
        return False
    return all(
        (p.is_file() and p.stat().st_size > 0) or p.is_dir()
        for p in outputs
    )


def _run(
    cmd: list[str],
    step: str,
    dry_run: bool,
    env: Optional[dict] = None,
) -> bool:
    """
    Run a subprocess command for *step*.

    Returns True on success, False on failure.
    Logs the command, elapsed time, and exit code.
    """
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"  CMD: {cmd_str}")

    if dry_run:
        logger.info(f"  [dry-run] skipping execution")
        return True

    t0 = time.monotonic()
    run_env = os.environ.copy()
    # Force UTF-8 I/O in child processes — prevents cp1252 crashes on Windows
    run_env["PYTHONUTF8"] = "1"
    run_env["PYTHONIOENCODING"] = "utf-8"
    if env:
        run_env.update(env)

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=run_env,
    )
    elapsed = time.monotonic() - t0

    if result.returncode == 0:
        logger.info(f"  [OK] {step} done in {elapsed/60:.1f} min")
        return True
    else:
        logger.error(
            f"  [FAIL] {step} FAILED (exit code {result.returncode}) "
            f"after {elapsed/60:.1f} min"
        )
        return False


# -- Individual step runners ------------------------------------------------

def step1_download_ai(args, dry_run: bool) -> bool:
    """Download CIFAKE AI-generated dataset."""
    logger.info("[step1] Downloading AI datasets…")
    return _run(
        [_python(), str(SCRIPTS / "datasets" / "download_ai.py"),
         "--dataset", "cifake"],
        step="step1",
        dry_run=dry_run,
    )


def step2_download_real(args, dry_run: bool) -> bool:
    """Download COCO val2017 real photos."""
    logger.info("[step2] Downloading real datasets…")
    return _run(
        [_python(), str(SCRIPTS / "datasets" / "download_real.py"),
         "--dataset", "coco"],
        step="step2",
        dry_run=dry_run,
    )


def step3_train_embedding(args, dry_run: bool) -> bool:
    """Train EfficientNet-B0 OwnEmbeddingModel."""
    logger.info("[step3] Training embedding detector…")
    cmd = [
        _python(), str(SCRIPTS / "train_embedding.py"),
        "--epochs", str(args.embed_epochs),
        "--batch",  str(args.embed_batch),
    ]
    if args.embed_limit:
        cmd += ["--limit", str(args.embed_limit)]
    return _run(cmd, step="step3", dry_run=dry_run)


def step4_build_clip_db(args, dry_run: bool) -> bool:
    """Build CLIP centroid database."""
    logger.info("[step4] Building CLIP centroid database…")
    return _run(
        [_python(), str(SCRIPTS / "build_clip_database.py")],
        step="step4",
        dry_run=dry_run,
    )


def step5_build_centroids(args, dry_run: bool) -> bool:
    """Build OwnEmbedding centroid database."""
    logger.info("[step5] Building OwnEmbedding centroids…")
    return _run(
        [_python(), str(SCRIPTS / "build_centroids.py")],
        step="step5",
        dry_run=dry_run,
    )


def step6_extract_features(args, dry_run: bool) -> bool:
    """Extract 30-signal feature vectors to features.csv."""
    logger.info("[step6] Extracting feature vectors…")

    real_cnt, ai_cnt = _count_manifest_rows()
    available        = min(real_cnt, ai_cnt) * 2
    limit            = min(args.feature_limit, available) if available else args.feature_limit

    logger.info(
        f"  manifest: {real_cnt} real, {ai_cnt} AI images available. "
        f"Extracting up to {limit} (--feature-limit={args.feature_limit})."
    )

    cmd = [
        _python(), str(SCRIPTS / "extract_features.py"),
        "--limit",   str(limit),
        "--workers", str(args.workers),
        "--seed",    "42",
    ]
    if args.resume_features and FEATURES.exists():
        cmd.append("--resume")

    return _run(cmd, step="step6", dry_run=dry_run)


def step7_train_ensemble(args, dry_run: bool) -> bool:
    """Train XGBoost meta-model."""
    logger.info("[step7] Training XGBoost meta-model…")
    return _run(
        [_python(), str(SCRIPTS / "train_ensemble.py")],
        step="step7",
        dry_run=dry_run,
    )


# -- Validation report ------------------------------------------------------

def validation_report() -> None:
    """Print a summary of all model artefacts and their status."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Post-training validation report")
    logger.info("=" * 70)

    files = {
        "own_embedding_model.pt":  REFERENCE / "own_embedding_model.pt",
        "clip_database.pkl":       REFERENCE / "clip_database.pkl",
        "own_centroids.pkl":       REFERENCE / "own_centroids.pkl",
        "ensemble_xgb.pkl":        REFERENCE / "ensemble_xgb.pkl",
        "ensemble_results.json":   REFERENCE / "ensemble_results.json",
        "platt_params.json":       REFERENCE / "platt_params.json",
        "features.csv":            FEATURES,
    }

    for name, path in files.items():
        if path.exists():
            size_kb = path.stat().st_size // 1024
            logger.info(f"  [OK] {name:<35} {size_kb:>8} KB  at {path}")
        else:
            logger.warning(f"  [FAIL]  {name:<35}  NOT FOUND  (expected at {path})")

    # Parse ensemble results if available
    results_path = REFERENCE / "ensemble_results.json"
    if results_path.exists():
        try:
            import json
            with open(results_path) as fh:
                res = json.load(fh)
            logger.info("")
            logger.info("XGBoost results:")
            logger.info(f"  CV AUC: {res.get('cv_auc_mean', '?'):.4f} ± {res.get('cv_auc_std', '?'):.4f}")
            logger.info(f"  Test AUC: {res.get('test_auc', '?'):.4f}")
            logger.info(f"  Test F1:  {res.get('test_f1', '?'):.4f}")
            logger.info(f"  Test Acc: {res.get('test_accuracy', '?'):.4f}")
        except Exception as exc:
            logger.warning(f"Could not parse ensemble_results.json: {exc}")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Restart the backend server")
    logger.info("  2. Check GET /health — all models should show 'ok'")
    logger.info("  3. Run: pytest backend/tests/ -v -m 'not slow'")
    logger.info("=" * 70)


# -- Main -------------------------------------------------------------------

ALL_STEPS = ["step1", "step2", "step3", "step4", "step5", "step6", "step7"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VeriFile-X — full training pipeline orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # -- Pipeline control -------------------------------------------------
    parser.add_argument(
        "--start-at", metavar="STEP",
        choices=ALL_STEPS,
        default=None,
        help=(
            "Resume pipeline from this step (e.g. step3). "
            "All earlier steps are skipped even if their outputs are missing."
        ),
    )
    parser.add_argument(
        "--only", metavar="STEP",
        choices=ALL_STEPS,
        default=None,
        help="Run only this single step and exit.",
    )
    parser.add_argument(
        "--skip", metavar="STEP1,STEP2",
        default="",
        help="Comma-separated list of steps to skip.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    parser.add_argument(
        "--skip-done", action="store_true",
        help=(
            "Skip a step if its expected outputs already exist. "
            "Useful for re-running after a mid-pipeline crash."
        ),
    )

    # -- Hardware / training hyperparameters ------------------------------
    parser.add_argument(
        "--embed-epochs", type=int, default=20,
        help="Epochs for train_embedding.py. Default: 20 (RTX 4050).",
    )
    parser.add_argument(
        "--embed-batch", type=int, default=24,
        help="Batch size for train_embedding.py. Default: 24 (fits RTX 4050 6 GB).",
    )
    parser.add_argument(
        "--embed-limit", type=int, default=0,
        help="Per-class image limit for embedding training. 0=all.",
    )
    parser.add_argument(
        "--feature-limit", type=int, default=50000,
        help=(
            "Max images for extract_features.py (balanced: half real, half AI). "
            "Default: 50000. Reduce to 10000 for a quick test run."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel worker processes for feature extraction. Default: 4.",
    )
    parser.add_argument(
        "--resume-features", action="store_true",
        help=(
            "Pass --resume to extract_features.py so it appends to an "
            "existing features.csv instead of restarting from scratch."
        ),
    )

    args = parser.parse_args()

    # -- Header ------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("VeriFile-X — Training Pipeline")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python:  {sys.version.split()[0]}  |  OS: {platform.system()}")
    logger.info(f"Root:    {ROOT}")
    logger.info(f"Log:     {LOG_FILE}")
    logger.info("=" * 70)

    # -- Pre-flight checks -------------------------------------------------
    _check_gpu()
    _check_disk(min_gb=15.0)

    # -- Determine which steps to run --------------------------------------
    skip_set: set[str] = set(s.strip() for s in args.skip.split(",") if s.strip())

    if args.only:
        run_steps = [args.only]
    elif args.start_at:
        start_idx = ALL_STEPS.index(args.start_at)
        run_steps = ALL_STEPS[start_idx:]
    else:
        run_steps = ALL_STEPS[:]

    run_steps = [s for s in run_steps if s not in skip_set]

    if args.dry_run:
        logger.info("[dry-run] Steps that would execute:")
        total_eta = sum(STEP_ETA_MIN.get(s, 0) for s in run_steps)
        for s in run_steps:
            logger.info(
                f"  {s}: {STEP_NAMES[s]}  (~{STEP_ETA_MIN.get(s, '?')} min)"
            )
        logger.info(f"Estimated total: {total_eta} min ({total_eta/60:.1f} h)")
    else:
        logger.info(f"Steps to run: {run_steps}")
        total_eta = sum(STEP_ETA_MIN.get(s, 0) for s in run_steps)
        logger.info(
            f"Estimated total: {total_eta} min ({total_eta/60:.1f} h). "
            "Actual time depends on data size and hardware."
        )

    # -- Step dispatch -----------------------------------------------------
    step_fns = {
        "step1": step1_download_ai,
        "step2": step2_download_real,
        "step3": step3_train_embedding,
        "step4": step4_build_clip_db,
        "step5": step5_build_centroids,
        "step6": step6_extract_features,
        "step7": step7_train_ensemble,
    }

    pipeline_start = time.monotonic()
    failed_steps: list[str] = []

    for step in run_steps:
        logger.info("")
        logger.info(f"--- {step}: {STEP_NAMES[step]} ---")

        # -- Disk space check before each step ----------------------------
        if not args.dry_run:
            free_gb = _check_disk(min_gb=5.0)
            if free_gb < 3.0:
                logger.error(
                    f"Less than 3 GB free before {step}. Aborting to protect disk."
                )
                sys.exit(1)

        # -- Skip-if-done --------------------------------------------------
        if args.skip_done and _step_already_done(step):
            logger.info(f"  <-  Outputs already exist — skipping (--skip-done).")
            continue

        # -- Run the step --------------------------------------------------
        fn = step_fns[step]
        ok = fn(args, dry_run=args.dry_run)

        if not ok:
            failed_steps.append(step)
            logger.error(
                f"Step {step} failed. Pipeline aborted.\n"
                f"  Fix the error above, then re-run with:\n"
                f"    python scripts/retrain_all.py --start-at {step}"
            )
            break

    # -- Summary -----------------------------------------------------------
    elapsed_total = time.monotonic() - pipeline_start
    logger.info("")
    logger.info("=" * 70)
    if failed_steps:
        logger.error(f"Pipeline FAILED at: {failed_steps}")
        logger.info(f"Elapsed: {elapsed_total/60:.1f} min")
        logger.info(f"Full log: {LOG_FILE}")
        sys.exit(1)
    else:
        logger.info(
            f"Pipeline COMPLETE in {elapsed_total/60:.1f} min  "
            f"({timedelta(seconds=int(elapsed_total))})"
        )
        logger.info(f"Full log: {LOG_FILE}")
        if not args.dry_run:
            validation_report()


if __name__ == "__main__":
    main()
