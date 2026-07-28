"""
Tests for backend/core/model_integrity.py (F-14).
"""
import pytest


def test_no_pinned_hash_allows_load_with_warning(tmp_path, caplog):
    """No entry in known_hashes.json for this filename -> allowed, but
    with a warning (see module docstring: refusing to load with zero
    pinned hashes at all would break every deployment today)."""
    from backend.core.model_integrity import verify_integrity

    f = tmp_path / "unpinned.pkl"
    f.write_bytes(b"some content")
    verify_integrity(f, known_hashes={})  # must not raise


def test_matching_pinned_hash_allows_load(tmp_path):
    from backend.core.model_integrity import verify_integrity, _sha256_of

    f = tmp_path / "pinned.pkl"
    f.write_bytes(b"real, untampered content")
    real_hash = _sha256_of(f)

    verify_integrity(f, known_hashes={"pinned.pkl": real_hash})  # must not raise


def test_tampered_file_raises_integrity_error(tmp_path):
    """The core regression test: a file whose content no longer matches
    its pinned hash must fail closed, not silently load."""
    from backend.core.model_integrity import verify_integrity, ModelIntegrityError, _sha256_of

    f = tmp_path / "pinned.pkl"
    f.write_bytes(b"original content")
    original_hash = _sha256_of(f)

    f.write_bytes(b"TAMPERED CONTENT injected by attacker")

    with pytest.raises(ModelIntegrityError):
        verify_integrity(f, known_hashes={"pinned.pkl": original_hash})


def test_warning_is_logged_only_once_per_file(tmp_path, caplog):
    """Avoid log-spam: the same unpinned filename shouldn't warn on
    every single call within a process lifetime."""
    from backend.core.model_integrity import verify_integrity, _warned_files

    f = tmp_path / "repeat_warn_test.pkl"
    f.write_bytes(b"content")
    _warned_files.discard(f.name)

    import logging
    with caplog.at_level(logging.WARNING):
        verify_integrity(f, known_hashes={})
        verify_integrity(f, known_hashes={})
        verify_integrity(f, known_hashes={})

    warnings = [r for r in caplog.records if f.name in r.message]
    assert len(warnings) == 1
