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

**Forensic-Grade AI Image Detection**

<img src="frontend/logo2.png" width="400" alt="VeriFile-X Logo"><br>


[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20Site-22d3ee?style=for-the-badge)](https://abinaze.github.io/VeriFile-X)
[![API](https://img.shields.io/badge/API-HuggingFace%20Space-ff6b35?style=for-the-badge)](https://abinazebinoy-verifile-x-api.hf.space)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-341%20Passing-brightgreen?style=for-the-badge)](backend/tests/)
[![Version](https://img.shields.io/badge/Version-7.2.0-blue?style=for-the-badge)](backend/core/config.py)

**26 Detection Signals · 85–92% Accuracy · Court-Ready Reports**

</div>

---

## Overview

VeriFile-X is an open-source, production-grade forensic AI image detection platform. It analyzes any image using 26 independent signals drawn from current research in computer vision, image forensics, and digital signal processing. Rather than returning a single score, it produces a full forensic report explaining exactly which signals fired, how confident each is, and why the verdict was reached.

Designed for journalists, legal professionals, researchers, and security teams who need verifiable, explainable results.

---

## Detection Methods

| Method | Signals | What It Detects |
|--------|---------|-----------------|
| DIRE — Diffusion Reconstruction Error | 4 | Stable Diffusion, DALL-E 3, Midjourney, Firefly |
| CLIP Universal Detection | 3 | Any generator — zero-shot generalization |
| Statistical & Frequency Analysis | 6 | DCT artifacts, noise floor, spectral anomalies |
| ELA — Error Level Analysis | 4 | JPEG compression inconsistencies, local edits |
| PRNU Camera Fingerprinting | 2 | Missing sensor noise (absent in all AI images) |
| Covariance & Eigenvalue Analysis | 3 | Sensor physics correlation breakdown |
| Metadata Forensics | 2 | EXIF inconsistencies, synthetic metadata |
| Platform & Generator Attribution | 2 | Social media re-encoding, generator fingerprints |

---

## Accuracy

| Generator | Accuracy |
|-----------|----------|
| Stable Diffusion (all versions) | 85–92% |
| DALL-E 3 | 85–92% |
| Midjourney v5–v6 | 85–92% |
| Adobe Firefly | 82–88% |
| StyleGAN / StyleGAN2 | 88–94% |
| Unknown / unseen generators | 70–80% (zero-shot) |

---

## Core Capabilities

### Forensic Analysis
- Full 26-signal analysis with per-signal scores, confidence levels, and explanations
- Manipulation localization heatmap (Grad-CAM, patch-level)
- Generator attribution: Stable Diffusion, DALL-E 3, SDXL, Midjourney, StyleGAN
- Social media platform fingerprinting (Instagram, Twitter, Facebook, LinkedIn, WhatsApp)
- C2PA content credential verification
- Adversarial robustness testing against compression, blur, and noise attacks

### Real-Time Streaming
- Server-Sent Events endpoint streams each signal result as it completes
- Signal waterfall UI showing live per-signal progress

### Batch Processing
- Process up to 10 images in a single request
- Aggregate statistics, duplicate detection, and risk ranking

### Evidence Case Management
- Create named investigation cases and attach analysis results
- Full-text search, status tracking, and summary generation
- Append-only JSONL audit trail with SHA-256 hash chaining

### Report Export
- PDF, JSON, and CSV export formats
- SHA-256 integrity hash per report for tamper detection

### API Key Management
- Role-based access: admin, analyst, viewer
- Raw keys never stored — SHA-256 hashes only
- Key creation, revocation, and live verification

---

## API Reference

Interactive documentation available at `/docs` when the server is running.

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/analyze/image` | Full 26-signal forensic analysis |
| POST | `/api/v1/analyze/image/stream` | Real-time SSE stream of signal results |
| POST | `/api/v1/analyze/heatmap` | Manipulation localization heatmap |
| POST | `/api/v1/analyze/attribution` | AI generator attribution |
| POST | `/api/v1/analyze/platform` | Social media platform detection |
| POST | `/api/v1/analyze/c2pa` | C2PA content credential verification |
| POST | `/api/v1/analyze/robustness` | Adversarial robustness test |
| POST | `/api/v1/analyze/batch` | Batch analysis — up to 10 images |
| POST | `/api/v1/analyze/export/{fmt}` | Export as `pdf`, `json`, or `csv` |
| GET | `/api/v1/analyze/history` | Recent analysis history |
| GET | `/api/v1/analyze/stats` | Aggregate detection statistics |

### Case Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/cases/` | Create investigation case |
| GET | `/api/v1/cases/` | List cases |
| GET | `/api/v1/cases/search` | Full-text case search |
| GET | `/api/v1/cases/{id}` | Get case with all evidence |
| POST | `/api/v1/cases/{id}/evidence` | Attach evidence to case |
| DELETE | `/api/v1/cases/{id}` | Archive case |

### Keys and Observability

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/keys/` | Create API key (admin) |
| GET | `/api/v1/keys/` | List keys (admin) |
| DELETE | `/api/v1/keys/{id}` | Revoke key (admin) |
| GET | `/api/v1/keys/verify` | Verify own key |
| GET | `/health` | Health check |
| GET | `/api/v1/metrics` | System metrics |
| POST | `/api/v1/metrics/reset` | Reset counters (requires `X-Admin-Key`) |

---

## Supported Formats

| Format | MIME Type | Max Size |
|--------|-----------|----------|
| JPEG | image/jpeg | 10 MB |
| PNG | image/png | 10 MB |
| WebP | image/webp | 10 MB |

---

## Quick Start

```bash
git clone https://github.com/abinaze/VeriFile-X.git
cd VeriFile-X

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r backend/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

uvicorn backend.main:app --reload --port 8000
```

Open `frontend/index.html` in your browser.
API docs: `http://localhost:8000/docs`

For Docker and production setup, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Project Structure

```
VeriFile-X/
├── backend/
│   ├── api/routes/           # analyze, cases, keys, upload
│   ├── core/                 # config, cache, audit_log, logger
│   ├── services/             # 26 detection services + ensemble
│   ├── models/               # Pydantic request/response models
│   ├── tests/                # 341 tests across 38 modules
│   └── main.py
├── frontend/
│   └── index.html            # Single-file, no build step required
├── .github/workflows/
│   ├── ci.yml                # Tests on every push
│   └── deploy-pages.yml      # GitHub Pages deployment
├── Dockerfile
├── DEPLOYMENT.md
└── PHASE_ROADMAP.md
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | localhost + production | Comma-separated allowed origins |
| `DEBUG` | `False` | Enable debug mode |
| `RATE_LIMIT_PER_MINUTE` | `10` | Global rate limit |
| `MAX_FILE_SIZE_MB` | `50` | Upload validation limit |
| `MAX_ANALYSIS_SIZE_MB` | `10` | Analysis processing limit |
| `CACHE_TTL_MINUTES` | `60` | Result cache duration |
| `ADMIN_KEY_HASH` | *(unset)* | SHA-256 hash of the admin key for metrics reset |

---

## Security

- Rate limiting on all endpoints (sliding window)
- IP SHA-256 hashing in logs (no raw IPs stored)
- Security headers: HSTS (HTTPS-only), CSP, X-Frame-Options, Permissions-Policy
- Input validation and injection detection on all routes
- Append-only hash-chained audit log
- API keys stored as SHA-256 hashes
- Admin key verified against `ADMIN_KEY_HASH` environment variable

See [SECURITY.md](SECURITY.md) for the responsible disclosure policy.

---

## Development

```bash
cd backend
pytest tests/ -v --tb=short
pytest tests/ --cov=. --cov-report=term-missing
flake8 backend/ --max-line-length=120
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*Developed and maintained by **Abinaze Binoy**.*
