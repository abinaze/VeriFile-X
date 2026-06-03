"""
Extract 30-signal feature vectors from all images in manifest.csv.

Usage:
    python scripts/extract_features.py
    python scripts/extract_features.py --limit 50000 --workers 4
    python scripts/extract_features.py --resume --limit 10000 --workers 6

What this does:
    1. Reads data/manifest.csv (train split only, to keep val/test clean)
    2. For each image, runs the full 30-signal VeriFile-X detection pipeline
    3. Extracts score from each signal into a flat feature row
    4. Saves all rows to data/features.csv
    5. Supports --resume to skip already-processed images (crash recovery)

Prerequisites (run in order):
    1. scripts/datasets/download_ai.py  --dataset cifake (or --dataset all)
    2. scripts/datasets/download_real.py --dataset coco  (or --dataset all)
    3. scripts/train_embedding.py        --epochs 20 --batch 24
    4. scripts/build_clip_database.py    --model ViT-B/32
    5. scripts/build_centroids.py
    6. THIS SCRIPT (extract_features.py)
    7. scripts/train_ensemble.py         --hparam-search

Hardware notes (RTX 4050, 6 GB VRAM):
    Use --workers 4 for CPU-bound multiprocessing.
    With 100 k images, expect 50–140 h; use --limit 50000 for a time-budgeted run.
"""
import sys
import csv
import time
import random
import logging
import argparse
import hashlib
import multiprocessing as mp
from pathlib import Path
from typing import Optional

# ── Project root on sys.path so backend imports work ──────────────────────
sys.path.insert(0, str(Path(__file__).parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
OUT_CSV  = ROOT / "data" / "features.csv"


# ── Worker function (top-level so multiprocessing can pickle it) ───────────

def _extract_one_worker(args_tuple) -> Optional[dict]:
    """
    Run the full 30-signal pipeline for a single image and return a flat
    feature row, or None on failure (caller writes a zero-filled fallback).

    This function is intentionally top-level (not a method or nested
    function) so Python's multiprocessing module can pickle it cleanly.
    """
    img_path_str, label_str, source_str = args_tuple

    # Import inside the worker so each process initialises its own copy of
    # the detector and avoids shared-memory CUDA / PyTorch state issues.
    try:
        from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    except ImportError as exc:
        return {"_error": f"import_failed: {exc}", "path": img_path_str,
                "label": label_str, "source": source_str}

    img_path = Path(img_path_str)
    if not img_path.exists():
        return None

    try:
        img_bytes = img_path.read_bytes()
        detector  = AdvancedEnsembleDetector(img_bytes, img_path.name)
        report    = detector.detect()
        detector.cleanup()

        row: dict = {
            "path":   img_path_str,
            "label":  1 if label_str == "ai" else 0,
            "source": source_str,
        }
        for sig in report.get("all_signals", []):
            col      = sig["signal_name"].lower().replace(" ", "_")
            row[col] = round(float(sig.get("score", 0.0)), 6)
        return row

    except Exception as exc:
        # Return a sentinel dict so the caller can log and zero-fill.
        return {"_error": str(exc), "path": img_path_str,
                "label": label_str, "source": source_str}


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_manifest(split: str = "train") -> list[dict]:
    """Load rows from manifest.csv for the given split."""
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found at {MANIFEST}.\n"
            "Run scripts/datasets/download_ai.py and download_real.py first."
        )
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("split", "train") == split:
                rows.append(row)
    logger.info(f"Loaded {len(rows)} rows from manifest (split={split!r})")
    return rows


def _sample_balanced(rows: list[dict], limit: int, seed: int = 42) -> list[dict]:
    """Randomly sample up to limit/2 real and limit/2 AI rows."""
    rng = random.Random(seed)
    real_rows = [r for r in rows if r.get("label") == "real"]
    ai_rows   = [r for r in rows if r.get("label") == "ai"]
    rng.shuffle(real_rows)
    rng.shuffle(ai_rows)
    half     = limit // 2
    sample   = real_rows[:half] + ai_rows[:half]
    rng.shuffle(sample)
    logger.info(
        f"Balanced sample: {min(len(real_rows), half)} real + "
        f"{min(len(ai_rows), half)} AI = {len(sample)} total"
    )
    return sample


def _load_already_processed() -> set[str]:
    """Return the set of absolute path strings already in features.csv."""
    done: set[str] = set()
    if not OUT_CSV.exists():
        return done
    try:
        with open(OUT_CSV, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                p = row.get("path", "")
                if p:
                    done.add(p)
        logger.info(f"Resume mode: {len(done)} rows already in {OUT_CSV}")
    except Exception as exc:
        logger.warning(f"Could not read existing features.csv: {exc}")
    return done


def _infer_fieldnames_from_row(row: dict) -> list[str]:
    """Determine canonical column order: path, label, source, then all signals."""
    meta = ["path", "label", "source"]
    signal_cols = [k for k in row if k not in meta and not k.startswith("_")]
    return meta + sorted(signal_cols)


def _zero_filled_row(img_path_str: str, label_str: str,
                     source_str: str, fieldnames: list[str]) -> dict:
    """Return a row with all signal columns set to 0.0 (extraction failed)."""
    row = {"path": img_path_str, "label": 1 if label_str == "ai" else 0,
           "source": source_str}
    for col in fieldnames:
        if col not in row:
            row[col] = 0.0
    return row


# ── Core extraction loop ───────────────────────────────────────────────────

def run_extraction(
    sample:    list[dict],
    workers:   int,
    resume:    bool,
    batch_log: int = 100,
) -> list[dict]:
    """
    Run feature extraction over *sample*.

    When workers > 1, uses a multiprocessing Pool so CPU-bound signal
    computations parallelise across cores.  Each worker imports the
    backend independently so PyTorch / CUDA state is never shared.
    """
    already_done = _load_already_processed() if resume else set()

    # Build work items, resolving paths relative to project root.
    work_items: list[tuple[str, str, str]] = []
    for row in sample:
        rel = row.get("path", "").replace("\\", "/")
        abs_path = str(ROOT / rel) if not Path(rel).is_absolute() else rel
        if resume and abs_path in already_done:
            continue
        work_items.append((abs_path, row.get("label", ""), row.get("source", "")))

    skipped = len(sample) - len(work_items)
    if skipped:
        logger.info(f"Resuming: skipping {skipped} already-processed images")

    if not work_items:
        logger.info("Nothing to process (all images already extracted).")
        return []

    logger.info(
        f"Extracting features from {len(work_items)} images "
        f"using {workers} worker(s)…"
    )

    results:    list[dict] = []
    fieldnames: Optional[list[str]] = None
    failed      = 0
    t0          = time.monotonic()

    def _handle_result(result: Optional[dict], item_tuple: tuple) -> None:
        nonlocal fieldnames, failed
        abs_path, label_str, source_str = item_tuple

        if result is None:
            # Image not found on disk — skip silently.
            return

        if "_error" in result:
            failed += 1
            logger.warning(
                f"  ✗ {Path(abs_path).name}: {result['_error']}"
            )
            # Zero-fill so the feature matrix is consistent.
            if fieldnames is not None:
                results.append(
                    _zero_filled_row(abs_path, label_str, source_str, fieldnames)
                )
            return

        # Good result.
        if fieldnames is None:
            fieldnames = _infer_fieldnames_from_row(result)

        # Ensure all columns present (handles signals that aren't always returned).
        for col in (fieldnames or []):
            if col not in result:
                result[col] = 0.0

        results.append(result)

    if workers <= 1:
        # ── Single-process mode ─────────────────────────────────────────
        for idx, item in enumerate(work_items, 1):
            result = _extract_one_worker(item)
            _handle_result(result, item)
            if idx % batch_log == 0:
                elapsed = time.monotonic() - t0
                rate    = idx / elapsed
                eta_s   = (len(work_items) - idx) / max(rate, 1e-6)
                logger.info(
                    f"  {idx}/{len(work_items)}  "
                    f"ok={len(results)}  failed={failed}  "
                    f"rate={rate:.1f}/s  ETA={eta_s/60:.1f}min"
                )
    else:
        # ── Multi-process mode ──────────────────────────────────────────
        # Use 'spawn' context on all platforms to avoid CUDA fork issues.
        ctx  = mp.get_context("spawn")
        pool = ctx.Pool(processes=workers)

        futures = [
            (item, pool.apply_async(_extract_one_worker, (item,)))
            for item in work_items
        ]
        pool.close()

        for idx, (item, future) in enumerate(futures, 1):
            try:
                result = future.get(timeout=120)   # 2-min per image max
            except mp.TimeoutError:
                logger.warning(f"  Timeout: {Path(item[0]).name}")
                result = {"_error": "timeout", "path": item[0],
                          "label": item[1], "source": item[2]}
            except Exception as exc:
                result = {"_error": str(exc), "path": item[0],
                          "label": item[1], "source": item[2]}

            _handle_result(result, item)

            if idx % batch_log == 0:
                elapsed = time.monotonic() - t0
                rate    = idx / elapsed
                eta_s   = (len(work_items) - idx) / max(rate, 1e-6)
                logger.info(
                    f"  {idx}/{len(work_items)}  "
                    f"ok={len(results)}  failed={failed}  "
                    f"rate={rate:.1f}/s  ETA={eta_s/60:.1f}min"
                )

        pool.join()

    elapsed = time.monotonic() - t0
    logger.info(
        f"Extraction complete: {len(results)} rows saved, "
        f"{failed} failed, {elapsed/60:.1f} min total"
    )
    return results


# ── CSV writer ─────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict], fieldnames: list[str], append: bool) -> None:
    """Write (or append) rows to features.csv."""
    mode    = "a" if append and OUT_CSV.exists() else "w"
    write_h = not (append and OUT_CSV.exists())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_h:
            writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"{'Appended' if mode == 'a' else 'Wrote'} "
        f"{len(rows)} rows to {OUT_CSV}"
    )


# ── Validation helper ──────────────────────────────────────────────────────

def _validate_output(fieldnames: list[str], n_rows: int) -> None:
    """Sanity-check the written CSV and log a summary."""
    if not OUT_CSV.exists() or n_rows == 0:
        logger.error("No features.csv written — something went wrong.")
        return

    try:
        with open(OUT_CSV, newline="", encoding="utf-8") as fh:
            reader      = csv.DictReader(fh)
            actual_cols = reader.fieldnames or []
            count       = sum(1 for _ in reader)

        signal_cols = [c for c in actual_cols if c not in ("path", "label", "source")]
        logger.info(
            f"✅ Validation: {count} rows, {len(signal_cols)} signal columns"
        )
        missing = [f for f in fieldnames if f not in actual_cols]
        if missing:
            logger.warning(f"Missing columns in output: {missing}")
    except Exception as exc:
        logger.warning(f"Validation error: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 30-signal feature vectors for XGBoost training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--limit", type=int, default=2000,
        help=(
            "Max images to process (balanced: limit/2 real + limit/2 AI). "
            "0 = use all. Default: 2000. For RTX-4050 time budget use 50000."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of parallel worker processes. "
            "Use 4–6 for a multi-core CPU. Default: 1 (single-process, "
            "easier to debug). Note: workers>1 uses 'spawn' context so "
            "each worker imports the backend fresh — avoids CUDA fork bugs."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Skip images already present in features.csv and append new rows. "
            "Useful after a crash or when extending an existing feature set."
        ),
    )
    parser.add_argument(
        "--split", default="train",
        help="Which manifest split to use. Default: 'train'.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for balanced sampling. Default: 42.",
    )
    parser.add_argument(
        "--log-every", type=int, default=100,
        help="Log progress every N images. Default: 100.",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("VeriFile-X — Feature Extraction")
    logger.info("=" * 70)
    logger.info(
        f"limit={args.limit or 'all'}  workers={args.workers}  "
        f"resume={args.resume}  split={args.split!r}  seed={args.seed}"
    )

    # 1. Load & sample manifest
    all_rows = _load_manifest(split=args.split)
    if not all_rows:
        logger.error(
            f"No rows found in manifest for split={args.split!r}. "
            "Run the download scripts first."
        )
        sys.exit(1)

    limit  = args.limit if args.limit > 0 else len(all_rows)
    sample = _sample_balanced(all_rows, limit, seed=args.seed)

    # 2. Extract features
    rows = run_extraction(
        sample    = sample,
        workers   = args.workers,
        resume    = args.resume,
        batch_log = args.log_every,
    )

    if not rows and not args.resume:
        logger.error(
            "No features extracted. Check that:\n"
            "  • Image files exist on disk (run download scripts)\n"
            "  • manifest.csv paths are correct\n"
            "  • backend imports are working (run from repo root)"
        )
        sys.exit(1)

    if not rows:
        logger.info("All images already processed (resume mode). Nothing to write.")
        return

    # 3. Determine canonical column order
    fieldnames = _infer_fieldnames_from_row(rows[0])

    # 4. Write CSV (append in resume mode, overwrite otherwise)
    _write_csv(rows, fieldnames=fieldnames, append=args.resume)

    # 5. Validate
    total_written = len(rows)
    if args.resume and OUT_CSV.exists():
        try:
            with open(OUT_CSV, newline="", encoding="utf-8") as fh:
                total_written = sum(1 for _ in csv.DictReader(fh))
        except Exception:
            pass

    _validate_output(fieldnames, total_written)
    logger.info("=" * 70)
    logger.info(f"Next step: python scripts/train_ensemble.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
