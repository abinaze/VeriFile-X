# Contributing to VeriFile-X

Contributions are welcome under the terms of the project's [license](LICENSE) (PolyForm Noncommercial 1.0.0) — by submitting a pull request, you agree your contribution is licensed under the same terms.

## Before You Start

For anything beyond a small fix (typo, docstring, obvious bug), open an issue first describing what you want to change and why. This avoids duplicated effort and lets us agree on the approach before you invest time in an implementation.

## Development Setup

```bash
git clone https://github.com/abinaze/VeriFile-X.git
cd VeriFile-X

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r backend/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

git checkout -b feature/your-feature-name
```

## Code Standards

- **Style** — PEP 8, enforced by `flake8` in CI (`.flake8` sets the project's line-length and ignore rules). Run `flake8 backend/` before committing.
- **Types** — new code should carry type hints; `mypy` runs in CI (non-blocking during the current cleanup period — see [PHASE_ROADMAP.md](PHASE_ROADMAP.md), Phase 33). Please don't introduce *new* mypy errors even while old ones are still being worked through.
- **Logging** — use `from backend.core.logger import setup_logger; logger = setup_logger(__name__)`, not bare `logging.getLogger`. Every module in the codebase now follows this consistently; new code should too.
- **Comments** — explain *why* a piece of logic exists or a specific constant was chosen, not what the code visibly already does.
- **No silent fallbacks** — if a signal can't run (missing model, corrupt input, unsupported format), return an explicit `confidence: 0.0` result rather than a best-guess placeholder score. This is what makes the confidence-gated ensemble in `advanced_ensemble_detector.py` work correctly; a signal that quietly returns a plausible-looking number instead of admitting it didn't run will get weighted as if it had.

## Testing

Every new feature or bug fix needs a test.

```bash
cd backend
pytest tests/ -v -m "not slow" --tb=short
pytest tests/ --cov=. --cov-report=term-missing
```

Coverage must not decrease. If you're fixing a bug, add a test that fails before your fix and passes after it — a regression test that only ever passes doesn't prove anything.

## Verifying a Patch Actually Applied

This codebase has, more than once, had a fix described in a commit message that didn't actually land in the file — a patch script's `assert` failed silently and the intended change was never written. Before committing:

```bash
python -m py_compile path/to/changed_file.py   # syntax check — proves nothing on its own
grep -n "<something that only exists in your new code>" path/to/changed_file.py   # proves the change is actually there
```

A clean compile is not evidence a change landed; an *unchanged* file also compiles cleanly. Grep for something that can only be true after your edit.

## Commit Messages

One logical change per commit. Explain what was broken (if it's a fix), why it mattered, and what the change does — not just "fix bug" or "update file."

## Pull Request Process

1. Rebase on the latest `main` before opening the PR
2. Update `README.md` and/or `PHASE_ROADMAP.md` if your change affects documented behavior, configuration, or the API surface
3. Ensure CI is green (`pytest`; `flake8`/`mypy`/`pip-audit` currently run non-blocking but should not show *new* errors introduced by your change)
4. Describe what changed and why in the PR description — link the issue it addresses
5. Request review; be responsive to feedback

## Reporting Bugs vs. Reporting Security Issues

Functional bugs: open a GitHub issue.

Security vulnerabilities: do **not** open a public issue — see [SECURITY.md](SECURITY.md) for the private disclosure process.

## Questions

Open an issue with the `question` label.
