---
title: VeriFile-X API
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---



<div align="center">

# VeriFile-X

**Forensic-Grade AI Image Detection Platform**

<img src="frontend/logo2.png" width="400" alt="VeriFile-X Logo"><br>


[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20Site-22d3ee?style=for-the-badge)](https://abinaze.github.io/VeriFile-X)
[![API](https://img.shields.io/badge/API-HuggingFace%20Space-ff6b35?style=for-the-badge)](https://abinazebinoy-verifile-x-api.hf.space)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-488%20Passing-brightgreen?style=for-the-badge)](backend/tests/)
[![Version](https://img.shields.io/badge/Version-8.5.0-blue?style=for-the-badge)](backend/core/config.py)
[![Coverage](https://img.shields.io/badge/Coverage-80%25-green?style=for-the-badge)](backend/tests/)

**30 Detection Signals · Ed25519 Signed Reports · MCMC Confidence Intervals · Court-Ready Forensics**

</div>

---

## Overview

VeriFile-X is an open-source, production-grade forensic AI image detection platform. It analyzes any image using **30 independent detection signals** drawn from published research in computer vision, image forensics, digital signal processing, and deep learning. Rather than returning a single opaque score, it produces a full forensic report explaining exactly which signals fired, their individual confidence levels, and why the verdict was reached.

Designed for **journalists, legal professionals, researchers, and security teams** who require verifiable, explainable, auditable results — not a black-box percentage.

---

## Honest Accuracy Disclosure

VeriFile-X accuracy depends on whether reference model files exist in `data/reference/`:

| Configuration | Signals Active | Realistic Accuracy |
|---|---|---|
| Full deployment (all models built) | 30/30 | 82-91% |
| No reference models (fresh install) | ~19/30 | 55-68% |
| CLIP only missing | 29/30 | 70-80% |

After a fresh `git clone`, the `data/` directories contain only `.gitkeep` files. You must build the reference models before deploying. See [Building Reference Models](#building-reference-models) below.

The `/health` endpoint reports exactly which models are loaded and the expected accuracy tier.

---

## Detection Architecture

### Signal Pipeline (30 Signals)

```
Image Input (JPEG / PNG / WebP / TIFF / HEIC)
        |
        v  EXIF orientation correction
        |
        +-> Statistical Analysis (19 sub-signals)
        |     FFT radial spectrum, KL divergence, Mahalanobis distance,
        |     DCT kurtosis, wavelet energy, GLCM texture, noise residual,
        |     spectral entropy, LBP texture, edge statistics, color correlation,
        |     compression artifacts, perturbation stability, eigenvalue spread,
        |     local covariance, patch anisotropy, color distribution,
        |     blockiness (inter-block), frequency ratio
        |
        +-> Deep Learning Signals
        |     DIRE (SD 2.1 diffusion reconstruction error)
        |     CLIP (embedding centroid distance)
        |     OwnEmbedding (fine-tuned EfficientNet-B0)
        |
        +-> Camera Forensics Signals
        |     PRNU (noise autocorrelation residual)
        |     CFA (Bayer demosaicing correlations)
        |     Noiseprint (Haar camera fingerprint)
        |
        +-> Compression Forensics Signals
        |     ELA (JPEG re-compression error level)
        |     JPEG Ghost (NSE energy curve Q=30-99)
        |     Noise Map (Gaussian residual frequency)
        |
        +-> Metadata Signal
              EXIF forensics (GPS plausibility, device fingerprint)
                       |
                       v
              Confidence-Gated Dynamic Ensemble
              (inactive signals excluded -- no 0.5 pollution)
                       |
                       +-> Platt Scaling Calibration
                       +-> MCMC Posterior Distribution
                       +-> XGBoost Meta-Model (when ensemble_xgb.pkl exists)
                                 |
                                 v
                       Forensic Report
                       ai_probability, classification, confidence
                       probability_distribution (interval_90, std, certainty)
                       chain_of_custody (Ed25519 signed, SHA-256 digest)
                       30 signal breakdowns with per-signal explanations
```

### Ensemble Weighting (DIRE-available branch)

| Signal | Weight |
|--------|--------|
| Statistical (19 sub-signals) | 0.26 |
| DIRE | 0.21 |
| CLIP | 0.16 |
| OwnEmbedding (when loaded) | 0.08 |
| PRNU | 0.08 |
| ELA | 0.07 |
| Metadata | 0.06 |
| DCT Frequency | 0.05 |
| JPEG Ghost | 0.04 |
| Noiseprint | 0.03 |
| Noise Map | 0.02 |
| CFA Bayer | 0.02 |
| **Total** | **1.00** |

Signals with `confidence=0` are excluded and remaining weights renormalised. A missing CLIP database does not pollute the score with a fixed 0.5.

---

## Building Reference Models

Without reference models the system runs on statistical signals only (~55-68% accuracy). Build them once after setup:

```bash
# 1. Collect datasets (minimum: 5,000 real + 5,000 AI images)
python scripts/download_real_images.py
python scripts/generate_ai_samples.py

# 2. Build CLIP reference database
python scripts/build_clip_database.py

# 3. Train OwnEmbedding model
python scripts/train_embedding.py --epochs 20 --batch 32
python scripts/build_centroids.py

# 4. Train XGBoost meta-model + Platt calibration
python scripts/train_ensemble.py

# 5. Verify
curl http://localhost:8000/health | python3 -m json.tool
```

Free datasets: CIFAKE (60k real + 60k AI), ArtiFact (2.5M, 13 generators), GenImage (1.35M, 8 generators)

---

## Supported Formats

| Format | MIME Type | Notes |
|--------|-----------|-------|
| JPEG | image/jpeg | Full signal support |
| PNG | image/png | ELA and JPEG Ghost return neutral (lossless) |
| WebP | image/webp | ELA and JPEG Ghost return neutral (lossless) |
| TIFF | image/tiff | Full signal support |
| HEIC | image/heic | Requires pillow-heif installed |
| HEIF | image/heif | Requires pillow-heif installed |

Max file size: 10 MB per analysis request.

---

## API Reference

Interactive docs: `/docs` (Swagger UI) and `/redoc`

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/analyze/image` | Full 30-signal forensic analysis |
| POST | `/api/v1/analyze/image/stream` | Real-time SSE stream per signal |
| POST | `/api/v1/analyze/segment` | Per-tile AI probability grid |
| POST | `/api/v1/analyze/heatmap` | Grad-CAM manipulation heatmap |
| POST | `/api/v1/analyze/attribution` | AI generator attribution |
| POST | `/api/v1/analyze/platform` | Social media platform detection |
| POST | `/api/v1/analyze/c2pa` | C2PA content credential verification |
| POST | `/api/v1/analyze/robustness` | Adversarial robustness test |
| POST | `/api/v1/analyze/batch` | Batch analysis (up to 10 images) |
| POST | `/api/v1/analyze/export/{fmt}` | Export: pdf, json, csv |
| GET | `/api/v1/analyze/history` | Recent analysis history |
| GET | `/api/v1/analyze/stats` | Aggregate statistics |

### Verification

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/verify/{evidence_id}` | Verify Ed25519 report signature (no auth) |
| GET | `/api/v1/verify/public-key` | Ed25519 public key PEM (no auth) |

### Case Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/cases/` | Create investigation case |
| GET | `/api/v1/cases/` | List cases |
| GET | `/api/v1/cases/search` | Full-text search |
| GET | `/api/v1/cases/{id}` | Get case with evidence |
| POST | `/api/v1/cases/{id}/evidence` | Attach evidence |
| DELETE | `/api/v1/cases/{id}` | Archive case |

### Webhooks and Feedback

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/webhooks/` | Register webhook (admin) |
| GET | `/api/v1/webhooks/` | List webhooks (admin) |
| DELETE | `/api/v1/webhooks/{id}` | Delete webhook (admin) |
| POST | `/api/v1/feedback/` | Submit analyst correction |
| GET | `/api/v1/feedback/` | Feedback history |
| GET | `/api/v1/feedback/weights` | Adaptive signal weights (admin) |

### Observability

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health + detector model status |
| GET | `/api/v1/metrics` | System metrics |
| POST | `/api/v1/metrics/reset` | Reset counters (admin) |

---

## Report Format

```json
{
  "evidence_id": "uuid5-from-sha256",
  "file_hash": "sha256:...",
  "summary": {
    "ai_probability": 0.87,
    "ai_classification": "likely_ai_generated",
    "confidence": "high",
    "total_detection_signals": 30,
    "suspicious_detection_signals": 22
  },
  "probability_distribution": {
    "point_estimate": 0.87,
    "interval_90": [0.74, 0.96],
    "interval_50": [0.81, 0.92],
    "std": 0.06,
    "certainty": "high"
  },
  "chain_of_custody": {
    "signed_at": "2026-05-27T10:00:00Z",
    "digest_sha256": "...",
    "signature": "base64url...",
    "algorithm": "Ed25519",
    "verify_url": "/api/v1/verify/..."
  },
  "detection_signals": [...]
}
```

### Classification Labels

| Label | Score Range | Meaning |
|-------|-------------|---------|
| `likely_ai_generated` | > 0.70 | Strong AI indicators |
| `possibly_ai_generated` | 0.60-0.70 | Moderate AI indicators |
| `inconclusive` | 0.40-0.60 | Conflicting or weak signals |
| `possibly_authentic` | 0.30-0.40 | Moderate authentic indicators |
| `likely_authentic` | < 0.30 | Strong authentic indicators |

---

## Quick Start

```bash
git clone https://github.com/abinaze/VeriFile-X.git
cd VeriFile-X

python -m venv venv
source venv/bin/activate

pip install -r backend/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

uvicorn backend.main:app --reload --port 8000
```

Open `frontend/index.html` in your browser. API docs: `http://localhost:8000/docs`

After starting, check `/health` to see which detector models are loaded.

For Docker and production deployment, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Project Structure

```
VeriFile-X/
+-- backend/
|   +-- api/routes/         analyze, cases, feedback, keys, segment, verify, webhooks
|   +-- core/               config, cache, audit_log, logger
|   +-- services/           30 detection signals + ensemble + MCMC + Platt + signing
|   +-- tests/              488 tests, 80%+ coverage
|   +-- utils/              validators, image_quality
|   +-- main.py
+-- data/
|   +-- reference/          Model files (built by scripts/ -- required for full accuracy)
|   +-- audit_log.jsonl     Append-only hash-chained audit trail
|   +-- cases.jsonl         Investigation cases
|   +-- feedback.jsonl      Analyst corrections
|   +-- signal_weights.json Nash adaptive weight overrides
+-- frontend/
|   +-- index.html          Single-file SPA, no build step required
+-- scripts/                Dataset collection + model training scripts
+-- .github/workflows/      CI (pytest + coverage) + GitHub Pages deployment
+-- Dockerfile
+-- DEPLOYMENT.md
+-- PHASE_ROADMAP.md
+-- SECURITY.md
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | localhost + production | Comma-separated allowed origins |
| `DEBUG` | `False` | Enables admin access without ADMIN_KEY_HASH (dev only) |
| `RATE_LIMIT_PER_MINUTE` | `10` | Rate limit per IP |
| `MAX_FILE_SIZE_MB` | `50` | Upload limit |
| `MAX_ANALYSIS_SIZE_MB` | `10` | Analysis processing limit |
| `CACHE_TTL_MINUTES` | `60` | Result cache TTL |
| `ADMIN_KEY_HASH` | required in production | SHA-256 of admin key. Unset + DEBUG=False blocks all admin endpoints. |

---

## Security

- Content-Length pre-check before reading file into RAM
- EXIF orientation applied before analysis (prevents spatial signal errors on rotated images)
- Rate limiting on all endpoints (sliding window per IP)
- Security headers: HSTS, CSP, X-Frame-Options, Permissions-Policy
- API keys stored as SHA-256 hashes only
- Ed25519 signed reports with public verification endpoint
- Admin endpoints blocked (503) when ADMIN_KEY_HASH unset in production
- Append-only hash-chained audit log
- Webhook HMAC-SHA256 signing

See [SECURITY.md](SECURITY.md) for responsible disclosure policy.

---

## Known Limitations

| Limitation | Notes |
|-----------|-------|
| CLIP/XGBoost require training data | Build reference models for full accuracy |
| DIRE requires SD 2.1 (~4 GB, GPU recommended) | Gracefully excluded when unavailable |
| PRNU is single-image autocorrelation, not true multi-image PRNU | Weighted conservatively |
| CFA breaks for iPhone Night Mode / pixel binning | Known physics limitation |
| Platt calibration uses defaults before fitting | Run train_ensemble.py |

---

## Development

```bash
cd backend
pytest tests/ -v -m "not slow" --tb=short
pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=70
flake8 backend/ --max-line-length=120
```

---

## License

MIT -- see [LICENSE](LICENSE).

---

*Developed and maintained by **Abinaze Binoy**.*
