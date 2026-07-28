"""
Generate data/reference/known_hashes.json from the real reference/model
files currently on disk (F-14).

Run this once you have real (non-Git-LFS-stub) copies of:
    data/reference/own_embedding_model.pt
    data/reference/clip_database.pkl
    data/reference/own_centroids.pkl
    data/reference/ensemble_xgb.pkl

...then commit the resulting known_hashes.json. From that point on,
backend/core/model_integrity.py's verify_integrity() fail-closes on any
mismatch instead of just logging a warning -- no code changes needed,
it reads this file automatically.

Usage:
    python scripts/generate_model_hashes.py
"""
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
REF_DIR = REPO_ROOT / "data" / "reference"
HASHES_PATH = REF_DIR / "known_hashes.json"

# Keep in sync with backend/main.py's _model_files list.
TRACKED_FILES = [
    "own_embedding_model.pt",
    "clip_database.pkl",
    "own_centroids.pkl",
    "ensemble_xgb.pkl",
]

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    hashes = {}
    if HASHES_PATH.exists():
        try:
            hashes = json.loads(HASHES_PATH.read_text(encoding="utf-8"))
        except Exception:
            hashes = {}

    skipped = []
    updated = []
    for fname in TRACKED_FILES:
        path = REF_DIR / fname
        if not path.exists():
            skipped.append((fname, "not found"))
            continue
        with open(path, "rb") as f:
            head = f.read(len(_LFS_POINTER_PREFIX))
        if head == _LFS_POINTER_PREFIX:
            skipped.append((fname, "still a Git-LFS pointer stub -- run `git lfs pull` first"))
            continue
        hashes[fname] = sha256_of(path)
        updated.append(fname)

    HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    HASHES_PATH.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for fname in updated:
        print(f"  hashed: {fname}")
    for fname, reason in skipped:
        print(f"  skipped: {fname} ({reason})")
    print(f"\nWrote {HASHES_PATH} with {len(hashes)} entries.")
    if skipped:
        print(
            f"\n{len(skipped)} file(s) were skipped -- those files will still "
            f"load with only a warning (no integrity check) until you re-run "
            f"this script after obtaining the real files."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
