# Testing Status

> **This file was previously badly out of date** (106 tests, v6.0.0, 21 signals — from an early
> phase of the project). Numbers below reflect the actual current test suite; re-verify them
> yourself with the commands at the bottom rather than trusting any static document indefinitely,
> including this one.

## Current status (v8.5.0)

- **Test functions defined:** 531 (`grep -rhE "^\s*def test_" backend/tests/*.py | wc -l`)
- **Test items collected by pytest:** ~544 (parametrized tests count each parameter set as a
  separate item, so this is higher than the raw function count)
- **Signal count:** 30 (19 statistical + 8 classical forensic + 3 deep-learning — see
  [`ARCHITECTURE.md`](ARCHITECTURE.md))

## Two-tier CI structure

`.github/workflows/ci.yml` splits tests into two jobs:

| Job | Marker | Timeout | On failure |
|---|---|---|---|
| Fast (required) | `-m "not slow"` | 60s per test | Blocks merge |
| Slow (advisory) | `-m "slow"` | 300s per test | `continue-on-error: true` — does not block merge |

`@pytest.mark.slow` is for tests that genuinely need real DIRE/CLIP model loading (multi-GB
download, no caching between CI runs) — `test_clip_database.py`, `test_clip_detector.py`,
`test_dire_detector.py`, and one test in `test_robustness.py`.

A `pytest_sessionstart` hook in `backend/tests/conftest.py` pre-warms the process-wide model cache
once, before any individual test's timeout clock starts — this exists because DIRE/CLIP/
OwnEmbedding load into a shared singleton cache, and without pre-warming, whichever full-pipeline
test happens to run first (of ~12 files that trigger the same load indirectly, not just the ones
directly marked `slow`) would absorb the entire cold-load cost inside the fast tier's 60-second
budget. If you add a new test that constructs `AdvancedEnsembleDetector` or calls
`generate_forensic_report()`, you don't need to do anything extra for this — the pre-warm hook
covers it automatically.

## Known environmental gaps (not code bugs)

Some sandboxed/restricted environments (no GPU, limited disk, or no network to the CLIP/HuggingFace
CDN) will show a handful of failures in `test_clip_database.py`, `test_clip_detector.py`,
`test_dire_detector.py`, and `test_own_embedding_detector.py` that trace back to the environment,
not the code — e.g. `ModuleNotFoundError: No module named 'clip'` or a blocked download. Before
treating any such failure as a real regression, confirm it also fails on an untouched baseline
checkout in the same environment.

## Run tests yourself

```bash
cd backend

# Fast tier only (matches CI's required, blocking job)
pytest tests/ -v -m "not slow" --tb=short --timeout=60

# Slow tier (real DIRE/CLIP model loading)
pytest tests/ -v -m "slow" --tb=short --timeout=300

# Everything, with coverage
pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=70

# Style / type checks
flake8 backend/ --max-line-length=120 --exclude=backend/tests/,backend/__pycache__
mypy backend/
```
