---
title: VeriFile-X API
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
<div align="center">
<h1>VeriFile-X</h1>

<img src="frontend/logo2.png" width="400" alt="VeriFile-X Logo"><br>

**A 30-signal ensemble platform for detecting AI-generated images, with fully explainable, per-signal forensic reports.**

[![License: Custom Noncommercial](https://img.shields.io/badge/License-Custom%20Noncommercial-2d3748?style=for-the-badge)](LICENSE.md)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-530%2B%20Passing-2f855a?style=for-the-badge)](backend/tests/)
[![Coverage](https://img.shields.io/badge/Coverage-~82%25-2f855a?style=for-the-badge)](backend/tests/)
[![Version](https://img.shields.io/badge/Version-8.5.0-2b6cb0?style=for-the-badge)](backend/core/config.py)

[Live Demo](https://abinaze.github.io/VeriFile-X) · [API Docs](https://abinazebinoy-verifile-x-api.hf.space/docs) · [Report an Issue](../../issues) · [Security Policy](SECURITY.md)

</div>

---

> **This is a demo, not a validated production forensics tool.** The learned components (the
> fine-tuned classifier and the reference-database signals) were trained on a small dataset, and no
> validated real-world accuracy figure exists for this deployment. Full detail, including a
> specific known data-leakage caveat on the one training run available, is in
> [Accuracy, Validation, and Honest Limitations](#accuracy-validation-and-honest-limitations) —
> read it before treating any output from the live demo as a reliable verdict, and never as the
> sole basis for a legal, journalistic, or moderation decision.

---

## Contents

- [Overview](#overview)
- [Why This Exists](#why-this-exists)
- [Core Capabilities](#core-capabilities)
- [Detection Architecture](#detection-architecture)
- [Ensemble Weighting](#ensemble-weighting)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Building Reference Models](#building-reference-models)
- [Full Step-by-Step Setup Guide](SETUP_GUIDE.md)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Usage Example](#usage-example)
- [Supported Formats](#supported-formats)
- [Report Format](#report-format)
- [Testing](#testing)
- [Security Model](#security-model)
- [Accuracy, Validation, and Honest Limitations](#accuracy-validation-and-honest-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

VeriFile-X analyzes a single uploaded image and returns a full forensic breakdown of whether it is authentic or AI-generated, built from **30 independently-computed detection signals** spanning:

- **Classical image forensics** — sensor-noise heuristics, JPEG/ELA compression-error analysis, JPEG-ghost double-compression detection, CFA (color filter array) demosaicing correlation, EXIF metadata plausibility checks
- **Frequency-domain statistics** — DCT coefficient analysis, FFT radial spectrum slope, wavelet energy, Mahalanobis distance and KL-divergence outlier detection against a natural-image prior, local covariance and patch-anisotropy variance
- **Deep-learning detectors** — a diffusion-reconstruction-error model (DIRE), CLIP zero-shot embedding-centroid distance, and a custom fine-tuned EfficientNet-B0 classifier

Rather than returning one opaque probability, every analysis reports **which signals fired, at what confidence, and why** — combined through a confidence-gated ensemble, with an **MCMC-derived posterior distribution** around the final estimate (a 90% credible interval and a certainty label) rather than a single unexplained point score.

The system is built API-first on FastAPI, with a lightweight single-page frontend, investigation case management, API-key based role access control, outbound webhook delivery, adversarial-robustness self-testing, and PDF/JSON/CSV report export.

## Why This Exists

Most publicly available "AI image detector" tools return a single percentage with no way to inspect how it was produced. That is close to useless for anyone who has to justify a conclusion — a journalist verifying a source photo, a moderator reviewing a report, a researcher benchmarking detection methods, or simply a developer who wants to understand *why* a model called something suspicious.

VeriFile-X is built around the opposite default: every score is decomposable. If you disagree with a verdict, you can see exactly which of the 30 signals drove it, at what individual confidence, and reproduce the computation yourself, because every signal is a documented, independently testable function rather than a black box.

## Core Capabilities

- **30-signal detection ensemble** across five detector families, described in [Detection Architecture](#detection-architecture) below
- **MCMC-calibrated confidence** — a Metropolis-Hastings sampler over the active signal set produces a full posterior distribution (point estimate, 50%/90% credible intervals, standard deviation, categorical certainty), not just a point score
- **Confidence-gated ensemble combination** — a signal that could not run (missing reference database, lossless format with no JPEG history, tiny image) is excluded and remaining weights are renormalized, rather than silently pulling the score toward a placeholder value
- **Analyst feedback loop** — corrections submitted through the feedback endpoint adjust per-signal weight multipliers for future analyses of similar inputs
- **Investigation case management** — group multiple pieces of evidence under a named case, with status tracking, tagging, and full-text search
- **Role-based API key access control** — admin / analyst roles, keys stored as salted hashes only, enforced uniformly across every router via a single shared dependency
- **Outbound webhooks** — HMAC-SHA256-signed delivery of completed analyses to a downstream URL, with automatic retry and an SSRF-hardened registration check (private/loopback/link-local/reserved/multicast destinations are rejected)
- **Batch and streaming analysis** — up to 10 images per batch request, or a single image streamed signal-by-signal over Server-Sent Events for real-time UI feedback
- **Adversarial robustness self-test** — re-runs detection after JPEG recompression, Gaussian blur/noise, downscale-upscale, and histogram equalization to report how stable a verdict is under common image transformations
- **Segment-level localization** — a per-tile probability grid for detecting partial AI insertion (an authentic background with an AI-generated subject composited in)
- **Generator attribution and platform-of-origin detection** — best-effort classification of which model family likely produced an image, and which social platform's re-encoding pipeline it passed through
- **C2PA content-credential scanning** and **PDF/JSON/CSV export** with a from-scratch, dependency-free PDF writer

## Detection Architecture

```
Image Input (JPEG / PNG / WebP / TIFF / HEIC / HEIF)
        |
        v  EXIF orientation correction, quality gate, extension/MIME cross-check
        |
        +-> Statistical Analysis (19 sub-signals)
        |     FFT radial spectrum, KL divergence, Mahalanobis distance,
        |     DCT kurtosis, wavelet energy, GLCM texture, noise residual,
        |     spectral entropy, LBP texture, edge statistics, color correlation,
        |     compression artifacts, perturbation stability, eigenvalue spread,
        |     local covariance, patch anisotropy, color distribution,
        |     inter-block regularity, radial frequency ratio
        |
        +-> Deep Learning Signals
        |     DIRE            diffusion reconstruction error (Stable Diffusion 2.1)
        |     CLIP             zero-shot embedding-centroid distance
        |     OwnEmbedding     fine-tuned EfficientNet-B0 classifier
        |
        +-> Camera / Sensor Forensics Signals
        |     PRNU-style noise heuristic (single-image; see limitations)
        |     CFA              Bayer demosaicing correlation
        |     Noiseprint       smoothed noise-residual patch consistency
        |
        +-> Compression Forensics Signals
        |     ELA              JPEG re-compression error level
        |     JPEG Ghost       double-compression energy curve
        |     Noise Map        Gaussian-residual frequency and regularity
        |
        +-> Metadata Signal
        |     EXIF forensics (signed GPS plausibility, device fingerprint,
        |     editing-software disclosure)
        |
        v
  Confidence-Gated Ensemble Combination
  (inactive signals excluded; remaining weights renormalized to 1.0;
   analyst-feedback weight multipliers applied)
        |
        +-> Platt-Scaling Calibration
        +-> MCMC Posterior Distribution (point estimate, credible intervals, certainty)
        +-> XGBoost meta-model override, when a trained model is present
        |
        v
  Forensic Report
  classification, calibrated probability, posterior distribution,
  per-signal breakdown with individual scores/confidence/explanations
```

Detector composition: `AdvancedEnsembleDetector` composes a `StatisticalDetector` (itself composing the 19-signal statistical bundle) alongside the standalone deep-learning and forensic-signal modules. An earlier inheritance-based design (`AdvancedAIDetector` -> `UltraAdvancedDetector` -> `CovarianceDetector` -> `StatisticalDetector`) was replaced by this composition-based structure; the three intermediate classes were later retired entirely once confirmed to have zero remaining production use (H-5).

## Ensemble Weighting

| Signal | Weight |
|---|---|
| DIRE (diffusion reconstruction) | 0.21 |
| Statistical analysis (19 sub-signals) | 0.20 |
| CLIP embedding distance | 0.14 |
| OwnEmbedding (fine-tuned EfficientNet-B0) | 0.08 |
| PRNU-style noise heuristic | 0.08 |
| ELA compression analysis | 0.07 |
| Metadata / EXIF forensics | 0.06 |
| DCT frequency artifacts | 0.05 |
| JPEG Ghost | 0.04 |
| Noiseprint | 0.03 |
| Noise Map | 0.02 |
| CFA Bayer correlation | 0.02 |
| **Total** | **1.00** |

This is a single, unified weighting path — an earlier version of the ensemble applied confidence-gating and feedback-weighting only when the DIRE model was unavailable, which meant the higher-accuracy DIRE-enabled deployment silently missed both refinements. That branch split has been removed: any signal with `confidence == 0` (a missing reference database, a lossless-format skip, a tiny/corrupt input) is excluded from the sum entirely, and the remaining weights are renormalized so they always total 1.0.

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| API framework | FastAPI 0.139, Pydantic v2, pydantic-settings |
| ASGI server | Uvicorn |
| Deep learning | PyTorch 2.7 (CPU build), torchvision, Hugging Face `diffusers` + `transformers`, OpenAI CLIP |
| Classical ML | scikit-learn, XGBoost, SHAP |
| Image processing | Pillow, OpenCV, scikit-image, pillow-heif (HEIC/HEIF) |
| Rate limiting | slowapi |
| Testing | pytest, pytest-cov, pytest-asyncio |
| Frontend | Single-file vanilla JavaScript SPA, no build step |
| CI | GitHub Actions (pytest + coverage, flake8, mypy, pip-audit) |
| Deployment | Docker, Hugging Face Spaces (API), GitHub Pages (frontend) |

## Project Structure

```
VeriFile-X/
├── backend/
│   ├── api/routes/       analyze, cases, feedback, keys, upload, webhooks
│   ├── core/             config, auth, cache, logger, audit_log, model_cache
│   ├── services/         30 detection signals + ensemble, MCMC, Platt calibration
│   ├── tests/            480+ tests, ~82% coverage
│   ├── utils/            validators, image quality gating
│   └── main.py
├── data/
│   ├── reference/        model/reference files (built via scripts/, gitignored)
│   ├── audit_log.jsonl   append-only analysis audit trail
│   ├── cases.jsonl        investigation case store
│   └── signal_weights.json  analyst-feedback weight overrides
├── frontend/
│   └── index.html        single-file SPA, no build step required
├── scripts/               dataset collection and model training scripts
├── assets/                logo and other repository media
├── .github/workflows/     CI (pytest, flake8, mypy, pip-audit) + Pages deploy
├── Dockerfile
├── DEPLOYMENT.md
├── PHASE_ROADMAP.md
├── SECURITY.md
├── CONTRIBUTING.md
└── LICENSE
```

## Getting Started

> This section is the quick path. For a genuinely step-by-step walkthrough — including *why* each
> step matters, how to verify it worked, and how to build a real (not placeholder) training
> dataset — see [`SETUP_GUIDE.md`](SETUP_GUIDE.md).

### Prerequisites

- Python 3.11 or later
- ~4 GB free disk space if you intend to run the DIRE detector locally (Stable Diffusion 2.1 weights); the system runs without it, with DIRE excluded from the ensemble and remaining weights renormalized
- Git, and [Git LFS](https://git-lfs.com/) if you want the real trained model files — a plain `git clone` or GitHub's "Download ZIP" only gives you LFS pointer stubs (harmless: the app degrades gracefully without them; see [`SETUP_GUIDE.md`](SETUP_GUIDE.md) to train your own)

### Installation

```bash
git clone https://github.com/abinaze/VeriFile-X.git
cd VeriFile-X

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r backend/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

uvicorn backend.main:app --reload --port 8000
```

Open `frontend/index.html` directly in a browser, or serve it with any static file server. Interactive API documentation is available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

After starting the server, check `GET /health` to see which detector models are actually loaded — the platform degrades gracefully (excluding a signal and renormalizing weights) rather than failing when a heavy model isn't available.

For Docker and production deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Building Reference Models

> The commands below reproduce this repo's own placeholder reference data quickly — useful for
> seeing the pipeline work end-to-end, but `scripts/generate_ai_samples.py` (step 1) generates
> *synthetic* placeholder images, not real AI-generator output, and won't itself produce a
> classifier worth trusting. [`SETUP_GUIDE.md`](SETUP_GUIDE.md) covers both this fast path and the
> real one (building an actually-sized, real dataset) in full, with explanations at each step.

Without built reference models, the ensemble runs on the statistical and metadata signals only, with CLIP, OwnEmbedding, and the XGBoost meta-model excluded and their weights renormalized across the remaining signals.

```bash
# 1. Collect datasets
python scripts/download_real_images.py
python scripts/generate_ai_samples.py

# 2. Build the CLIP reference database
python scripts/build_clip_database.py

# 3. Train the OwnEmbedding model
python scripts/train_embedding.py --epochs 20 --batch 32
python scripts/build_centroids.py

# 4. Train the XGBoost meta-model and Platt calibration
python scripts/train_ensemble.py

# 5. Verify
curl http://localhost:8000/health | python3 -m json.tool
```

See [Accuracy, Validation, and Honest Limitations](#accuracy-validation-and-honest-limitations) before treating any locally-trained meta-model's reported metrics as a validated accuracy figure — `scripts/train_ensemble.py` includes its own data-leakage self-check for exactly this reason.

## Configuration

All settings are read via `pydantic-settings` in `backend/core/config.py`, with the following environment variables:

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | localhost + production origins | Comma-separated allowed origins |
| `DEBUG` | `false` | Enables verbose logging; does not weaken the admin gate |
| `ADMIN_KEY_HASH` | *(required)* | SHA-256 hash of the admin key. Admin endpoints return 503 if unset, unless `ALLOW_INSECURE_ADMIN=true` is explicitly set for local development |
| `ALLOW_INSECURE_ADMIN` | `false` | Local-development-only opt-in to a length-only admin check when `ADMIN_KEY_HASH` is not configured. Never set this in production |
| `RATE_LIMIT_PER_MINUTE` | `10` | Default per-IP rate limit, applied as the fallback for any endpoint without its own explicit, more specific limit |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |
| `MAX_ANALYSIS_SIZE_MB` | `10` | Size limit for the analysis pipeline specifically |
| `CACHE_TTL_MINUTES` | `60` | Result cache time-to-live |
| `MAX_CACHE_SIZE` | `500` | Maximum cached result entries |
| `LOG_LEVEL` | `INFO` | Log level when `DEBUG` is false |

## API Reference

Interactive, always-current documentation: `/docs` (Swagger UI) and `/redoc`. The tables below summarize the endpoint surface; treat `/docs` as authoritative for exact request/response schemas.

### Analysis (`/api/v1/analyze`) — requires an analyst or admin API key

| Method | Endpoint | Description |
|---|---|---|
| POST | `/image` | Full 30-signal forensic analysis |
| POST | `/image/stream` | Real-time Server-Sent Events stream, one event per signal |
| POST | `/segment` | Per-tile AI-probability grid |
| POST | `/heatmap` | Grad-CAM manipulation-localization heatmap |
| POST | `/attribution` | Generator attribution (which model family likely produced the image) |
| POST | `/platform` | Social-platform re-encoding fingerprint detection |
| POST | `/c2pa` | C2PA content-credential scan |
| POST | `/robustness` | Adversarial robustness self-test |
| POST | `/batch` | Batch analysis, up to 10 images |
| POST | `/export/{format}` | Export a report as `pdf`, `json`, or `csv` |
| GET | `/history` | Recent analysis history |
| GET | `/stats` | Aggregate statistics |

### Case Management (`/api/v1/cases`) — requires an analyst or admin API key

| Method | Endpoint | Description |
|---|---|---|
| POST | `/` | Create an investigation case |
| GET | `/` | List cases |
| GET | `/search` | Full-text case search |
| GET | `/{id}` | Get a case with its attached evidence |
| POST | `/{id}/evidence` | Attach evidence to a case |
| DELETE | `/{id}` | Archive a case |

### Keys, Webhooks, and Feedback

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/keys/` | admin | Create an API key |
| GET | `/api/v1/keys/` | admin | List API keys |
| DELETE | `/api/v1/keys/{id}` | admin | Revoke an API key |
| GET | `/api/v1/keys/verify` | any valid key | Verify the caller's own key |
| POST | `/api/v1/webhooks/` | admin | Register a webhook (hostname resolved and checked against private/loopback/reserved ranges) |
| GET | `/api/v1/webhooks/` | admin | List webhooks |
| DELETE | `/api/v1/webhooks/{id}` | admin | Delete a webhook |
| POST | `/api/v1/feedback/` | analyst/admin | Submit an analyst correction |
| GET | `/api/v1/feedback/` | analyst/admin | Feedback history |
| GET | `/api/v1/feedback/weights` | admin | Current adaptive signal-weight overrides |

### Observability

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check plus which detector models are currently loaded |
| GET | `/api/v1/metrics` | Aggregate system metrics |
| POST | `/api/v1/metrics/reset` | Reset metrics counters (admin only) |

## Usage Example

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/image" \
  -H "Authorization: Bearer <your-analyst-key>" \
  -F "file=@photo.jpg"
```

## Supported Formats

| Format | MIME Type | Notes |
|---|---|---|
| JPEG | `image/jpeg` | Full signal support |
| PNG | `image/png` | ELA and JPEG Ghost return a neutral, non-participating score (lossless format, no prior JPEG history) |
| WebP | `image/webp` | Same lossless-format handling as PNG |
| TIFF | `image/tiff` | Full signal support |
| HEIC / HEIF | `image/heic`, `image/heif` | Requires `pillow-heif` |

Maximum file size: see `MAX_ANALYSIS_SIZE_MB` in [Configuration](#configuration).

## Report Format

Every analysis returns a JSON report keyed by a stable `evidence_id` (a UUID5 derived from the file's SHA-256 hash, so re-analyzing the same file always yields the same ID), containing file/EXIF metadata, a tampering analysis, the full per-signal breakdown, and a summary classification with an MCMC-derived probability distribution. See `/docs` for the exact, generated schema — response shapes are intentionally not duplicated here in prose to avoid this document drifting out of sync with the code.

### Classification Labels

| Label | Typical Score Range | Meaning |
|---|---|---|
| `likely_ai_generated` | > 0.70 | Strong AI indicators |
| `possibly_ai_generated` | 0.60 – 0.70 | Moderate AI indicators |
| `inconclusive` | 0.40 – 0.60 | Conflicting or weak signals |
| `possibly_authentic` | 0.30 – 0.40 | Moderate authentic indicators |
| `likely_authentic` | < 0.30 | Strong authentic indicators |

## Testing

```bash
cd backend
pytest tests/ -v -m "not slow" --tb=short
pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=70
flake8 backend/ --max-line-length=120
mypy backend/
```

CI runs the same suite on every push and pull request; see `.github/workflows/ci.yml`.

## Security Model

- API-key based role access control (admin / analyst), enforced uniformly across every router through a single shared FastAPI dependency, not a per-file reimplementation
- Keys are stored as salted hashes only; the raw key is shown exactly once, at creation
- Admin endpoints fail closed by default: they return 503 unless `ADMIN_KEY_HASH` is configured (or `ALLOW_INSECURE_ADMIN=true` is explicitly set for local development)
- Webhook URLs are resolved and checked against private, loopback, link-local, reserved, and multicast IP ranges before registration, to prevent SSRF
- Per-IP sliding-window rate limiting on every endpoint
- Security response headers: HSTS, CSP, X-Frame-Options, Permissions-Policy
- User-supplied strings (filename, EXIF fields) are HTML-escaped before rendering in the frontend
- An append-only, timestamped audit log records filename, hash, and verdict per analysis for accountability; this is a disclosed, deliberate feature, not incidental logging

See [SECURITY.md](SECURITY.md) for the responsible-disclosure process.

## Accuracy, Validation, and Honest Limitations

VeriFile-X's 480+ tests verify that the *pipeline* behaves correctly — bounded output ranges, determinism on repeated analysis of the same file, graceful degradation on corrupt or unusual input, and correct ensemble arithmetic. That is a meaningfully different claim from validated, real-world classification accuracy against a large, diverse, properly held-out set of authentic and AI-generated images, and this README does not publish a specific accuracy percentage, because the one internal training run available (`data/reference/ensemble_results.json`) evaluates on a small fraction of the dataset scale described in `data/DATASETS.md`, and `scripts/train_ensemble.py`'s own logging flags that specific result as showing signs consistent with data leakage rather than genuine cross-generator separability.

Re-validating the meta-model on a larger, resolution-matched dataset with grouped, leakage-checked cross-validation (splitting by source/generation run, not by individual image) is the top item on the [roadmap](PHASE_ROADMAP.md). Until that work lands, treat VeriFile-X as a rich, inspectable ensemble of individually-reasoned forensic signals with a documented methodology — not as a benchmark-proven or court-validated accuracy claim.

Other known, specific limitations:

| Limitation | Detail |
|---|---|
| CLIP / OwnEmbedding / XGBoost need built reference data | See [Building Reference Models](#building-reference-models); excluded gracefully (weights renormalized) when absent |
| DIRE requires Stable Diffusion 2.1 (~4 GB; GPU recommended) | Excluded gracefully when unavailable |
| The PRNU-style signal is a single-image heuristic | It is not multi-image camera-reference-pattern PRNU forensics, and is weighted and documented accordingly |
| CFA analysis can be unreliable on computational photography | Night-mode multi-frame fusion and pixel-binning sensors break the classical Bayer-pattern assumption |
| Platt calibration uses fixed defaults until fitted | Run `scripts/train_ensemble.py` against your own labeled holdout to fit it |
| C2PA scanning is header/XMP-based, not the reference C2PA SDK | Chosen for deployment portability; documented in the signal's own `accuracy_note` field |

## Roadmap

Full phase-by-phase history and forward-looking plans: [PHASE_ROADMAP.md](PHASE_ROADMAP.md).

## Contributing

Contributions are welcome under the terms of the project [License](#license) — see [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, code style, and pull request process.

## License

VeriFile-X is released under a custom license — see [LICENSE](LICENSE.md) for the full text.

In short: you're free to use, study, modify, and share this project — including its code, methodology, and any findings derived from it — for personal, academic, research, nonprofit, and any other non-commercial purpose. You may **not** use it, or anything built from it, for business or profit-making purposes without the author's prior written permission, and you must keep the license and copyright notice intact and credit the original author — this work may not be presented as created by anyone else. For commercial licensing or any other permission request, contact **Abinaze Binoy** at **abinazebinoy@gmail.com**.

## Author

Developed and maintained by **Abinaze Binoy**.

Project: [github.com/abinaze/VeriFile-X](https://github.com/abinaze/VeriFile-X)
