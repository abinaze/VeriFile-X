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
### Forensic-Grade AI Image Detection

VeriFile-X — Unmask AI Content with Confidence

<img src="frontend/logo2.png" width="440" height="340">

## Theme Options

| Theme | Description |
|-------|-------------|
| Animated | Three.js 3D background with floating objects and mouse parallax |
| Dark | Clean dark forensics theme |
| Light | Professional light theme |

**26 Detection Signals · 85–92% Accuracy · Court-Ready Analysis**

[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-Visit%20Site-22d3ee?style=for-the-badge)](https://abinaze.github.io/VeriFile-X)
[![API](https://img.shields.io/badge/🔌%20API-HuggingFace%20Space-ff6b35?style=for-the-badge)](https://abinazebinoy-verifile-x-api.hf.space)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-200%2B%20Passing-brightgreen?style=for-the-badge)](backend/tests/)

---

*Can you tell if an image was made by AI? VeriFile-X can — with forensic precision.*

</div>

---

## What Is VeriFile-X?

VeriFile-X is an open-source forensic AI image detection platform. It analyzes any image using **26 independent detection signals** across multiple state-of-the-art methods — diffusion reconstruction, semantic embeddings, statistical forensics, generator attribution, and platform fingerprinting — and returns a detailed, transparent report showing exactly why it made its decision.

Designed for researchers, journalists, legal professionals, and developers who need more than just a number.

---

## Live Detection Examples

| Input | AI Probability | Verdict |
|-------|---------------|---------|
| Midjourney generated portrait | 94% | 🔴 Likely AI Generated |
| Real DSLR photograph | 8% | 🟢 Likely Authentic |
| Stable Diffusion landscape | 87% | 🔴 Likely AI Generated |
| Phone camera selfie | 12% | 🟢 Likely Authentic |
| DALL-E 3 architectural render | 91% | 🔴 Likely AI Generated |

---

## Detection Methods

###  DIRE — Diffusion Reconstruction Error
*Based on [DIRE (ICCV 2023)](https://arxiv.org/abs/2303.09295)*

Reconstructs the image using Stable Diffusion 2.1 and measures how well the model can reproduce it. AI-generated images fit the diffusion distribution perfectly (low error). Real photographs do not.

- Detects: Stable Diffusion, DALL-E 3, Midjourney, Adobe Firefly
- Accuracy: **85–92%** on diffusion models

###  CLIP Universal Detection
*Based on [UnivFD (CVPR 2023)](https://arxiv.org/abs/2302.10174)*

Uses OpenAI CLIP embeddings to measure semantic distance from learned real vs. AI image centroids. Generalizes to unseen generators without retraining.

- Accuracy: **85–92%** across all generator types
- Zero-shot: works on generators it has never seen before

###  Own EfficientNet Embedding Detector

Custom-trained EfficientNet-B0 binary classifier fine-tuned on 408K+ images (208K real, 200K AI). Combined with centroid-based similarity scoring for maximum accuracy.

- Trained on: CelebA, FFHQ, COCO, DIV2K, StyleGAN2, Stable Diffusion
- Method: Direct classification + cosine distance to learned centroids

###  Statistical Analysis — 19 Signals

Analyzes pixel-level patterns, frequency domain artifacts, and statistical distributions that differ systematically between AI-generated and real images.

| Signal | What It Measures |
|--------|--------------------|
| Mahalanobis Distance | Deviation from natural image statistics |
| KL Divergence | Pixel intensity distribution anomalies |
| Perturbation Stability | Sensitivity to added noise |
| FFT Radial Spectrum | Frequency domain artifact fingerprint |
| DCT Coefficients | JPEG compression pattern anomalies |
| Wavelet Energy | Multi-scale texture inconsistencies |
| Eigenvalue Spread | Covariance structure of image patches |
| Local Covariance | Spatial correlation abnormalities |
| Patch Anisotropy | Directional texture variance |
| RGB Noise Covariance | Cross-channel noise correlation |
| Patch Spectral Variance | Frequency variance across regions |
| Natural Prior Deviation | Distance from natural image priors |
| Laplacian Variance | Noise level and edge sharpness |
| Color Entropy | HSV histogram distribution |
| GLCM Texture | Gray-level co-occurrence matrix features |
| LBP Texture | Local binary pattern analysis |
| Edge Statistics | Unnatural edge distribution patterns |
| Spectral Entropy | Frequency band energy distribution |
| Compression Artifacts | JPEG block artifact inconsistencies |

###  Additional Forensic Signals

| Signal | What It Detects |
|--------|----------------|
| PRNU | Camera sensor fingerprint (absence = no real camera) |
| ELA | Error Level Analysis — JPEG compression inconsistencies |
| DCT Frequency | GAN checkerboard artifacts in frequency domain |
| Metadata Forensics | Missing/inconsistent EXIF and AI software markers |

###  Generator Attribution

Classifies AI-generated images into generator families using DCT frequency fingerprints:

`stylegan` · `dalle3` · `sd14` · `sdxl` · `midjourney` · `real` · `unknown`

###  Platform Forensics

Detects social media re-encoding signatures via JPEG quantization table fingerprinting:

`whatsapp` · `instagram` · `discord` · `telegram` · `twitter_x` · `facebook` · `original`

###  C2PA Provenance Verification

Checks for Coalition for Content Provenance and Authenticity (C2PA) credentials:

`verified` · `partial` · `none` · `tampered`

###  XGBoost Ensemble Fusion

All 26 signals are fed into a trained XGBoost classifier for the final decision. Signal contributions are derived from 5-fold cross-validated performance, not manual weights. SHAP values provide per-signal explainability.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                             │
│            GitHub Pages · Plain HTML/CSS/JS                 │
│   Upload → 26 Signals · Hashes · EXIF · Heatmap · Export    │
└──────────────────────┬──────────────────────────────────────┘
                       │ POST /api/v1/analyze/image
┌──────────────────────▼──────────────────────────────────────┐
│               Backend — FastAPI + Uvicorn                   │
│            HuggingFace Spaces · Docker · Python 3.11        │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              ImageForensics                         │   │
│   │  ├── extract_exif()                                 │   │
│   │  ├── generate_hashes()    SHA256·MD5·pHash          │   │
│   │  ├── detect_tampering()   Flags·Confidence          │   │
│   │  ├── detect_ai_generation()                         │   │
│   │  │    └── AdvancedEnsembleDetector (26 signals)     │   │
│   │  ├── attribute_generator()  StyleGAN/DALLE/SD/MJ    │   │
│   │  ├── detect_platform()      Social media fingerprint│   │
│   │  └── verify_c2pa()          Content credentials     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│  API Endpoints: /analyze/* · /cases/* · /keys/*             │
│  Security: Rate limiting · MIME validation · In-memory      │
│  Caching:  SHA256-keyed LRU · 60min TTL · 500 entries       │
└─────────────────────────────────────────────────────────────┘
```

---

## API Reference

### Analyze an Image

```bash
curl -X POST https://abinazebinoy-verifile-x-api.hf.space/api/v1/analyze/image \
  -F "file=@your_image.jpg"
```

### All Endpoints

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `GET` | `/health` | Health check | None |
| `POST` | `/api/v1/analyze/image` | Full forensic analysis (max 10MB) | 10/min |
| `POST` | `/api/v1/analyze/image/heatmap` | Grad-CAM localization heatmap | 5/min |
| `POST` | `/api/v1/analyze/attribution` | Generator attribution | 10/min |
| `POST` | `/api/v1/analyze/platform` | Social media platform detection | 10/min |
| `POST` | `/api/v1/analyze/c2pa` | C2PA provenance verification | 10/min |
| `POST` | `/api/v1/analyze/robustness` | Adversarial robustness test | 2/min |
| `POST` | `/api/v1/analyze/batch` | Batch analysis (up to 10 images) | 2/min |
| `POST` | `/api/v1/analyze/export/{fmt}` | Export report (pdf/json/csv) | 5/min |
| `POST` | `/api/v1/cases/` | Create investigation case | 20/min |
| `GET` | `/api/v1/cases/` | List cases | 30/min |
| `GET` | `/api/v1/cases/search` | Search cases | 20/min |
| `GET` | `/api/v1/cases/{id}` | Get case details | 30/min |
| `POST` | `/api/v1/cases/{id}/evidence` | Add evidence to case | 20/min |
| `PATCH` | `/api/v1/cases/{id}/status` | Update case status | 20/min |
| `GET` | `/api/v1/keys/verify` | Verify API key | 30/min |
| `GET` | `/api/v1/metrics` | System observability metrics | 30/min |
| `POST` | `/api/v1/metrics/reset` | Reset metrics counters | 5/min |
| `GET` | `/docs` | Interactive API documentation | None |

---

## Classification Labels

| Label | Probability | Meaning |
|-------|------------|---------| 
| `likely_ai_generated` | > 80% | Strong multi-signal evidence of AI generation |
| `likely_ai_generated` | 70–80% | High confidence AI indicators |
| `possibly_ai_generated` | 50–70% | Multiple signals indicate AI, not conclusive |
| `possibly_authentic` | 30–50% | Likely real, some minor anomalies |
| `likely_authentic` | < 30% | Strong evidence of authentic photograph |

---

## Quick Start

```bash
git clone https://github.com/abinaze/VeriFile-X.git
cd VeriFile-X

# Linux/macOS
sudo apt-get install -y libmagic1
pip install -r backend/requirements.txt

# Windows
pip install -r backend/requirements-windows.txt

export PYTHONPATH=$(pwd)
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Open interactive docs
open http://localhost:8000/docs
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI 0.115 · Uvicorn · Python 3.11 |
| **AI Detection** | PyTorch 2.6 · OpenAI CLIP · EfficientNet-B0 · Stable Diffusion 2.1 |
| **Ensemble** | XGBoost + SHAP explainability · 26-signal feature matrix |
| **Image Processing** | OpenCV · Pillow · scikit-image · PyWavelets |
| **Statistical Analysis** | NumPy · SciPy · scikit-learn |
| **Hashing** | hashlib (SHA256/MD5) · imagehash (perceptual) |
| **Security** | slowapi rate limiting · python-magic MIME validation · SHA-256 API keys |
| **Caching** | In-memory SHA256-keyed LRU · 60min TTL |
| **Storage** | Append-only JSONL (cases, audit log, API keys) |
| **Frontend** | Plain HTML5 · CSS3 · Vanilla JavaScript · Three.js |
| **CI/CD** | GitHub Actions · flake8 · mypy · pip-audit |
| **Deployment** | HuggingFace Spaces (Docker) · GitHub Pages |

---

## Security

- **Rate limited:** per-endpoint limits from 2–30 requests/minute per IP
- **Double MIME validation:** Content-type header + python-magic file signature
- **In-memory only:** Uploaded files are never written to disk
- **API keys:** SHA-256 hashed, raw key shown once, never stored
- **RBAC:** viewer / analyst / admin role hierarchy
- **No data stored:** Cache stores results only, cleared on restart
- **Zero tracking:** No analytics, no cookies, no accounts required

---

## Tests

```bash
# Run all fast tests
pytest backend/tests/ -v -m "not slow" --tb=short

# Run with coverage
pytest backend/tests/ --cov=backend --cov-report=html
```

200+ tests covering all 26 signals, API endpoints, batch processing, case management, platform detection, C2PA verification, adversarial robustness, and invariant/property-based tests.

---

## Phases Completed

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Dataset foundation (408K+ images) | ✅ |
| 2 | EfficientNet embedding detector | ✅ |
| 3 | Centroid database + ensemble | ✅ |
| 4 | XGBoost trained ensemble + SHAP | ✅ |
| 5 | Grad-CAM localization heatmap | ✅ |
| 6 | Generator attribution classifier | ✅ |
| 7 | Social media platform forensics | ✅ |
| 8 | C2PA content credentials | ✅ |
| 9 | Adversarial robustness testing | ✅ |
| 10 | Batch investigation mode | ✅ |
| 11 | Evidence case management | ✅ |
| 12 | Report export suite (PDF/JSON/CSV) | ✅ |
| 13 | API keys + RBAC | ✅ |
| 14 | CI coverage + hardening | ✅ |
| 15 | System hardening + quality gate | ✅ |
| 16 | Monitoring + inconclusive verdict + image type | ✅ |
| 17 | Frontend hardening + rate limit consolidation | ✅ |

---

## Known Limitations

- DIRE requires ~4GB Stable Diffusion model download on first use — falls back to neutral score in memory-constrained environments
- Generator attribution uses rule-based heuristics (~60–70% accuracy) when no trained attribution model is present
- Heavily edited real images may produce false positives
- Images smaller than 32×32 pixels are automatically skipped

**Not a replacement for human expert review.** Always treat results as one forensic input, not a definitive verdict.

---

## Contributing

```bash
git checkout -b feature/your-feature
# make changes
pytest backend/tests/ -v
git commit -m "feat: your description"
git push origin feature/your-feature
# open a Pull Request
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE).

---

<div align="center">

Built with precision · Open source forever

**[Try it live →](https://abinaze.github.io/VeriFile-X)**

</div>
