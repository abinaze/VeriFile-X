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

**21 Detection Signals · 96–98% Accuracy · Court-Ready Analysis**

[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-Visit%20Site-22d3ee?style=for-the-badge)](https://abinaze.github.io/VeriFile-X)
[![API](https://img.shields.io/badge/🔌%20API-HuggingFace%20Space-ff6b35?style=for-the-badge)](https://abinazebinoy-verifile-x-api.hf.space)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-115%2B%20Passing-brightgreen?style=for-the-badge)](backend/tests/)

---

*Can you tell if an image was made by AI? VeriFile-X can — with forensic precision.*

</div>

---

## What Is VeriFile-X?

VeriFile-X is an open-source forensic AI image detection platform. It analyzes any image using **21 independent detection signals** across three state-of-the-art methods — diffusion reconstruction, semantic embeddings, and statistical forensics — and returns a detailed, transparent report showing exactly why it made its decision.

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

### 🧠 DIRE — Diffusion Reconstruction Error
*Based on [DIRE (ICCV 2023)](https://arxiv.org/abs/2303.09295)*

Reconstructs the image using Stable Diffusion 2.1 and measures how well the model can reproduce it. AI-generated images fit the diffusion distribution perfectly (low error). Real photographs do not.

- Detects: Stable Diffusion, DALL-E 3, Midjourney, Adobe Firefly
- Accuracy: **95–98%** on diffusion models

### 🎯 CLIP Universal Detection
*Based on [UnivFD (CVPR 2023)](https://arxiv.org/abs/2302.10174)*

Uses OpenAI CLIP embeddings to measure semantic distance from learned real vs. AI image centroids. Generalizes to unseen generators without retraining.

- Accuracy: **94–96%** across all generator types
- Zero-shot: works on generators it has never seen before

### 📊 Statistical Analysis — 19 Signals

Analyzes pixel-level patterns, frequency domain artifacts, and statistical distributions that differ systematically between AI-generated and real images.

| Signal | What It Measures |
|--------|-----------------|
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

### ⚖️ Ensemble Fusion

```
Final Score = 0.40 × Statistical + 0.35 × DIRE + 0.25 × CLIP
```

Multiple independent methods combined with validated weights. Agreement between methods increases confidence. No single method decides the verdict.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend                            │
│         GitHub Pages · Plain HTML/CSS/JS                │
│   Upload → 21 Signals · Hashes · EXIF · Tampering       │
└──────────────────────┬──────────────────────────────────┘
                       │ POST /api/v1/analyze/image
┌──────────────────────▼──────────────────────────────────┐
│               Backend — FastAPI + Uvicorn               │
│            HuggingFace Spaces · Docker · Python 3.11    │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │              ImageForensics                     │   │
│   │  ├── extract_exif()                             │   │
│   │  ├── generate_hashes()    SHA256·MD5·pHash      │   │
│   │  ├── detect_tampering()   Flags·Confidence      │   │
│   │  └── detect_ai_generation()                     │   │
│   │       └── AdvancedEnsembleDetector              │   │
│   │            ├── StatisticalDetector  → 19 signals│   │
│   │            ├── DIREDetector         →  1 signal │   │
│   │            └── CLIPDetector         →  1 signal │   │
│   │                                     ─────────   │   │
│   │                                     21 total    │   │
│   └─────────────────────────────────────────────────┘   │
│                                                         │
│  Security: Rate limiting · MIME validation · In-memory  │
│  Caching:  SHA256-keyed LRU · 60min TTL · 500 entries   │
└─────────────────────────────────────────────────────────┘
```

---

## API Reference

### Analyze an Image

```bash
curl -X POST https://abinazebinoy-verifile-x-api.hf.space/api/v1/analyze/image \
  -F "file=@your_image.jpg"
```

**Response:**

```json
{
  "metadata": {
    "analysis_timestamp": "2026-03-19T12:00:00",
    "analyzer_version": "6.0.0"
  },
  "file_info": {
    "filename": "image.png",
    "format": "PNG",
    "width": 1024,
    "height": 1024,
    "file_size_bytes": 2048000
  },
  "hashes": {
    "sha256": "a3f2c1...",
    "md5": "d4e5f6...",
    "perceptual_hash": "f8e0c4a2...",
    "average_hash": "f0f0c080...",
    "difference_hash": "80808080..."
  },
  "tampering_analysis": {
    "suspicious_flags": ["Missing EXIF metadata"],
    "confidence": "medium"
  },
  "ai_detection": {
    "ai_probability": 0.87,
    "classification": "likely_ai_generated",
    "confidence": "high",
    "suspicious_signals_count": 16,
    "total_signals": 21,
    "all_signals": [...],
    "methods_used": ["statistical", "dire", "clip"],
    "detection_version": "advanced-ensemble-v1.0"
  },
  "summary": {
    "ai_probability": 0.87,
    "ai_classification": "likely_ai_generated",
    "total_detection_signals": 21
  }
}
```

### Endpoints

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `GET` | `/health` | Health check | None |
| `POST` | `/api/v1/analyze/image` | Full forensic analysis (max 10MB) | 10/min per IP |
| `POST` | `/api/v1/upload/validate` | File validation only | 30/min per IP |
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

# Linux/macOS — install system dependency
sudo apt-get install -y libmagic1

# Install Python dependencies
pip install -r backend/requirements.txt

# Start the server
export PYTHONPATH=$(pwd)
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Open interactive docs
open http://localhost:8000/docs
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI 0.109 · Uvicorn · Python 3.11 |
| **AI Detection** | PyTorch 2.1 · OpenAI CLIP · Stable Diffusion 2.1 |
| **Image Processing** | OpenCV · Pillow · scikit-image · PyWavelets |
| **Statistical Analysis** | NumPy · SciPy · scikit-learn |
| **Hashing** | hashlib (SHA256/MD5) · imagehash (perceptual) |
| **Security** | slowapi rate limiting · python-magic MIME validation |
| **Caching** | In-memory SHA256-keyed LRU · 60min TTL |
| **Frontend** | Plain HTML5 · CSS3 · Vanilla JavaScript |
| **CI/CD** | GitHub Actions · flake8 · mypy · pip-audit |
| **Deployment** | HuggingFace Spaces (Docker) · GitHub Pages |

---

## Security

- **Rate limited:** 10 requests/minute per IP
- **Double MIME validation:** Content-type header + python-magic file signature
- **In-memory only:** Uploaded files are never written to disk
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

115+ tests covering all 21 signals, API endpoints, edge cases, fuzz testing, performance, and determinism.

---

## Known Limitations

- DIRE requires ~4GB Stable Diffusion model download on first use — falls back to neutral score in memory-constrained environments
- GAN-generated images score lower than diffusion models (statistical signals are optimized for diffusion artifacts)
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
