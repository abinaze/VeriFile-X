# VeriFile-X Architecture

This document describes how a request actually flows through the current system. If you're new to
the codebase, read this alongside [`SETUP_GUIDE.md`](../SETUP_GUIDE.md), which walks through
building and running every piece described here from nothing.

> **This file was previously out of date** — an earlier version described a much simpler,
> statistics-only pipeline with a hardcoded 10MB limit and no deep-learning signals. That no longer
> matches the code. Everything below was checked against the actual source as of v8.5.0.

## System overview

VeriFile-X is a FastAPI backend plus a static single-page frontend. A client uploads one image; the
backend runs **30 independent detection signals** across five detector families, combines them
through a confidence-gated weighted ensemble, and returns a full forensic report — not just a
score, but which signals fired, at what confidence, and why.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Client (browser SPA, or curl / any HTTP client)                     │
└───────────────────────────────┬────────────────────────────────────┘
                                 │ HTTPS, multipart/form-data upload
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FastAPI app (backend/main.py)                                       │
│  • Security headers middleware (CSP, HSTS, X-Frame-Options, ...)     │
│  • CORS                                                              │
│  • Per-IP sliding-window rate limiting (slowapi)                     │
│  • API-key auth dependency (require_role_for_method / _or_demo)      │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Upload validation (backend/utils/validators.py)                     │
│  • MIME/magic-byte check, extension cross-check, size limit          │
│  • Decompression-bomb guard (checked BEFORE EXIF re-encode)          │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ImageForensics.generate_forensic_report()                           │
│  (backend/services/image_forensics.py)                               │
│                                                                        │
│  1. EXIF extraction + orientation normalize                          │
│  2. Hash generation (SHA-256, MD5, perceptual hash)                  │
│  3. Tampering indicators (editing-tool / AI-marker EXIF keyword scan)│
│  4. detect_ai_generation() → AdvancedEnsembleDetector.detect()       │
│  5. Generator attribution, platform-of-origin, C2PA scan             │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  AdvancedEnsembleDetector.detect()                                    │
│  (backend/services/advanced_ensemble_detector.py)                    │
│                                                                        │
│  Runs 9 mutually-independent, non-torch signals concurrently in a    │
│  ThreadPoolExecutor (capped to os.cpu_count()):                      │
│    • Statistical bundle (19 sub-signals, see below)                  │
│    • PRNU, ELA, Metadata, DCT, JPEG Ghost, Noise Map,                │
│      Noiseprint, CFA                                                 │
│                                                                        │
│  Then runs 3 deep-learning signals sequentially (deliberately not    │
│  yet parallelized — see PROFILING_F17.md):                           │
│    • DIRE     — diffusion reconstruction error (Stable Diffusion 2.1)│
│    • CLIP     — zero-shot embedding-centroid distance                │
│    • OwnEmbedding — fine-tuned EfficientNet-B0 classifier            │
│                                                                        │
│  30 signals total. combine_signals() applies confidence-gated,       │
│  static per-category weights (see table below), renormalizing when   │
│  a signal is excluded (confidence == 0 — e.g. lossless format with   │
│  no JPEG history, missing reference database, tiny image).           │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Calibration + uncertainty                                            │
│  • Platt-scaling calibration (backend/services/platt_calibrator.py)  │
│  • MCMC posterior (Metropolis-Hastings) — point estimate, 50%/90%    │
│    credible intervals, certainty label                               │
│  • XGBoost meta-model override, when a trained model is present      │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
                    Forensic report (JSON), keyed by a stable
                    evidence_id (UUID5 of the file's SHA-256 hash)
```

## The 19-signal statistical bundle

Since the F-16 refactor, this bundle is composition-based, not inheritance-based —
`backend/services/statistical_signals.py` defines four independent components sharing a read-only
`ImageContext` dataclass:

| Component | Signal count | What it covers |
|---|---|---|
| `BasicSignals` | 10 | FFT radial spectrum, DCT coefficients, wavelet energy, GLCM texture, LBP texture, noise residual, spectral entropy, edge statistics, color correlation, compression artifacts |
| `UltraSignals` | 3 | Perturbation stability, radial frequency ratio, inter-block regularity |
| `CovarianceSignals` | 3 | Local covariance, patch anisotropy variance, eigenvalue spread |
| `AdvancedSignals` | 3 | Mahalanobis distance, KL divergence, color distribution outlier detection |

`AdvancedAIDetector` / `UltraAdvancedDetector` / `CovarianceDetector` / `StatisticalDetector` — the
original four classes from before F-16 — still exist as thin, backward-compatible facades over
these components (kept because tests construct them directly), but the real production entry
point, `AdvancedEnsembleDetector`, composes a `StatisticalDetector` instance rather than inheriting
from it.

## Ensemble weighting

Static per-category weights, summing to exactly 1.00:

| Signal | Weight |
|---|---|
| DIRE (diffusion reconstruction) | 0.21 |
| Statistical bundle (19 sub-signals) | 0.20 |
| CLIP embedding distance | 0.14 |
| OwnEmbedding (EfficientNet-B0) | 0.08 |
| PRNU-style noise heuristic | 0.08 |
| ELA compression analysis | 0.07 |
| Metadata / EXIF forensics | 0.06 |
| DCT frequency artifacts | 0.05 |
| JPEG Ghost | 0.04 |
| Noiseprint | 0.03 |
| Noise Map | 0.02 |
| CFA Bayer correlation | 0.02 |

Confidence **gates inclusion** — a signal with `confidence == 0` is dropped and the remaining
weights renormalize to still sum to 1.0 — it does not scale a signal's contribution proportionally.
A signal with any nonzero confidence contributes its full static weight.

## Model caching and concurrency

`backend/core/model_cache.py` provides a process-wide singleton (`ModelCache`) that DIRE, CLIP, and
OwnEmbedding all load into, guarded by a per-key lock so concurrent cold-start requests don't each
independently trigger a full model load. Two concurrency-relevant details worth knowing if you're
extending this:

- **CLIP and OwnEmbedding** cache only frozen, eval-mode model weights — safe to share across
  concurrent requests, since inference-only forward passes don't mutate shared state.
- **DIRE** additionally caches a `DDIMScheduler`, which *is* mutable
  (`set_timesteps()`/`step()`/`add_noise()` all write instance state). Every `DIREDetector`
  instance clones a fresh scheduler from the cached one's config on every cache hit
  (`_clone_scheduler()`) rather than sharing the cached object directly — this was a real
  thread-safety bug, fixed after being flagged during F-17 profiling, not a defensive-for-its-own-
  sake pattern.

## Request/response shape

Every analysis is keyed by a stable `evidence_id` — a UUID5 derived from the file's SHA-256 hash,
so re-analyzing the same file always produces the same ID. The full, generated response schema is
authoritative at `/docs` (Swagger UI) and `/redoc`; it isn't duplicated here to avoid this document
drifting out of sync with the code the way its predecessor did.

## Where to go next

- **Setting this whole system up from nothing, including training the ML components**:
  [`SETUP_GUIDE.md`](../SETUP_GUIDE.md)
- **API endpoint reference**: [`README.md`](../README.md#api-reference)
- **Security model**: [`SECURITY.md`](../SECURITY.md)
- **Deployment**: [`DEPLOYMENT.md`](../DEPLOYMENT.md)
- **Development history, phase by phase**: [`PHASE_ROADMAP.md`](../PHASE_ROADMAP.md)
- **Signal latency profiling and the parallelization work**: [`PROFILING_F17.md`](../PROFILING_F17.md)
