"""
Model/reference-file integrity checking (F-14).

pickle.load() on a file whose provenance is "whatever a remote channel
currently contains" is a real code-execution surface if that channel is
ever compromised -- and main.py's startup step downloads
own_embedding_model.pt, clip_database.pkl, own_centroids.pkl, and
ensemble_xgb.pkl from a Hugging Face Space named by the SPACE_ID
environment variable, with no verification before those .pkl files are
unpickled. SPACE_ID is attacker-influenceable in a misconfigured
deployment.

This module provides a SHA-256 checksum gate in front of every
pickle.load() call site, checked against a value pinned in THIS repo
(KNOWN_HASHES below / data/reference/known_hashes.json) -- not fetched
from the same untrusted channel as the files themselves, which is the
whole point.

Current status (see REMAINING_FIXES.md): KNOWN_HASHES starts empty.
This session's uploaded snapshot only contains Git-LFS pointer stubs
for data/reference/*.pkl (132-byte text files, not real model data), so
there is nothing real to hash yet. Once real model files exist in your
environment, run:

    python scripts/generate_model_hashes.py

...to populate data/reference/known_hashes.json with real SHA-256
values, then commit that file. Until it's populated, verify_integrity()
logs a loud warning (once per file) instead of hard-failing -- refusing
to start the app with no pinned hashes at all would break every
deployment today, which is worse than the gap it would close. Once
real hashes are committed, this becomes fail-closed automatically with
no further code changes: any mismatch raises ModelIntegrityError.
"""
import hashlib
import json
from pathlib import Path
from typing import Optional

from backend.core.logger import setup_logger

logger = setup_logger(__name__)

_HASHES_PATH = Path(__file__).parent.parent.parent / "data" / "reference" / "known_hashes.json"
_warned_files: set = set()


class ModelIntegrityError(Exception):
    """Raised when a reference/model file's SHA-256 does not match the
    value pinned in data/reference/known_hashes.json."""


def _load_known_hashes() -> dict:
    if not _HASHES_PATH.exists():
        return {}
    try:
        with open(_HASHES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not parse {_HASHES_PATH}: {e}")
        return {}


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_integrity(path: Path, known_hashes: Optional[dict] = None) -> None:
    """Verify path's SHA-256 against the pinned value for its filename.

    - No pinned hash on file for this filename -> log a warning once,
      then allow the load (see module docstring for why this isn't a
      hard failure yet).
    - Pinned hash present and it matches -> silent success.
    - Pinned hash present and it does NOT match -> raises
      ModelIntegrityError. Callers should let this propagate (do not
      swallow it into the generic "file missing/corrupt, fall back to
      neutral" except-branches these call sites already have) since a
      hash mismatch is a materially different, higher-severity signal
      than "file absent" or "file is an LFS pointer stub".
    """
    known = known_hashes if known_hashes is not None else _load_known_hashes()
    expected = known.get(path.name)
    if not expected:
        if path.name not in _warned_files:
            _warned_files.add(path.name)
            logger.warning(
                "No pinned SHA-256 for %s in %s -- integrity NOT verified "
                "before loading. Run scripts/generate_model_hashes.py once "
                "you have real model files, then commit the result.",
                path.name, _HASHES_PATH,
            )
        return
    actual = _sha256_of(path)
    if actual != expected:
        raise ModelIntegrityError(
            f"{path.name}: SHA-256 mismatch. Expected {expected}, got {actual}. "
            f"Refusing to load -- this file's contents do not match what is "
            f"pinned in {_HASHES_PATH}."
        )
