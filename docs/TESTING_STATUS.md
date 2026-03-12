# Testing Status

## Current Status
- **Total Tests:** 106
- **Passing:** 104
- **Skipped:** 2
- **Failing:** 0

## Test Coverage

### Detector Signal Counts
- **Ensemble (Full System):** 21 signals
  - Statistical: 19 signals
  - DIRE: 1 signal
  - CLIP: 1 signal

- **Individual Detectors:**
  - StatisticalDetector: 19 signals
  - CovarianceDetector: 16 signals
  - UltraAdvancedDetector: 13 signals
  - AdvancedAIDetector: 10 signals

### Version
- **Current Version:** 6.0.0
- **Last Updated:** March 9, 2026

### Known Limitations
- DIRE detector requires HuggingFace authentication in CI (graceful fallback implemented)
- CLIP centroids are placeholder values until reference database is built (Issue #32)
- Expected variance in AI probability: ~12-15% due to CLIP randomness

### Test Performance
- Fast tests (no ML models): ~2 minutes
- Slow tests (ML model loading): ~3 minutes
- Total: ~5 minutes

Run fast tests only:
```bash
pytest backend/tests/ -m "not slow" -v
```

Run all tests:
```bash
pytest backend/tests/ -v
```
