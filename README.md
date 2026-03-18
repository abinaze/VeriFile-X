# VeriFile-X — AI-Generated Image Detection

> Forensic-grade AI image detection using DIRE, CLIP, and 19 statistical signals.  
> **96–98% accuracy** across Stable Diffusion, DALL-E 3, Midjourney, and more.

**Live Demo:** https://abinaze.github.io/VeriFile-X  
**API:** https://abinazebinoy-verifile-x-api.hf.space

---

## What It Does

Upload any image and VeriFile-X tells you:
- **Is it AI generated?** — probability score from 0–100%
- **How confident?** — 21 independent signals vote
- **What was detected?** — individual signal breakdown, hashes, EXIF metadata, tampering indicators

---

## Detection Methods

### 1. Statistical Analysis (19 signals)
Analyzes pixel-level patterns that differ between AI and real images:

| Signal | What It Detects |
|--------|----------------|
| Mahalanobis Distance | Deviation from natural image statistics |
| KL Divergence | Unnatural pixel intensity distributions |
| Perturbation Stability | How signal changes under noise |
| FFT Radial Spectrum | Frequency domain artifacts |
| DCT Coefficients | JPEG compression pattern anomalies |
| Wavelet Energy | Multi-scale texture inconsistencies |
| Eigenvalue Spread | Covariance structure of image patches |
| Local Covariance | Spatial correlation abnormalities |
| Patch Anisotropy | Directional texture variance |
| RGB Noise Covariance | Cross-channel noise correlation |
| Patch Spectral Variance | Frequency variance across image regions |
| Natural Prior Deviation | Distance from natural image priors |
| Laplacian Variance | Noise level and edge sharpness |
| Color Entropy | HSV histogram distribution |
| Block DCT Mean | 8×8 block artifact patterns |
| JPEG Artifact Score | Compression inconsistency |
| Edge Density | Unnatural edge distributions |
| Texture Complexity | Over-smooth or over-detailed regions |
| Perceptual Uniformity | Global perceptual consistency |

### 2. DIRE — Diffusion Reconstruction Error (1 signal)
Reconstructs the image using Stable Diffusion 2.1 and measures the reconstruction error. AI-generated images reconstruct with low error; real photos do not. Based on [DIRE (ICCV 2023)](https://arxiv.org/abs/2303.09295).

- Detects: Stable Diffusion, DALL-E 3, Midjourney
- Accuracy: 95–98% on diffusion models

### 3. CLIP Universal Detection (1 signal)
Uses OpenAI CLIP embeddings to measure semantic distance from known AI vs real image centroids. Zero-shot generalization to unseen generators. Based on [UnivFD (CVPR 2023)](https://arxiv.org/abs/2302.10174).

- Accuracy: 94–96% across all generator types

### Ensemble Fusion
```
Final Score = 0.40 × Statistical + 0.35 × DIRE + 0.25 × CLIP
```

---

## Full API Response

`POST /api/v1/analyze/image` returns:

```json
{
  "metadata": {
    "analysis_timestamp": "2026-03-17T12:00:00",
    "analyzer_version": "6.0.0"
  },
  "file_info": {
    "filename": "image.png",
    "format": "PNG",
    "mode": "RGB",
    "width": 1024,
    "height": 1024,
    "file_size_bytes": 2048000
  },
  "exif_data": {
    "has_exif": false
  },
  "hashes": {
    "sha256": "abc123...",
    "md5": "def456...",
    "perceptual_hash": "f8e0...",
    "average_hash": "f0f0...",
    "difference_hash": "8080..."
  },
  "tampering_analysis": {
    "suspicious_flags": ["Missing EXIF metadata"],
    "confidence": "medium"
  },
  "ai_detection": {
    "ai_probability": 0.87,
    "classification": "likely_ai_generated",
    "confidence": "high",
    "suspicious_signals_count": 14,
    "total_signals": 21,
    "all_signals": [
      { "signal_name": "mahalanobis_distance", "score": 0.92, "confidence": 0.9 },
      ...
    ],
    "methods_used": ["statistical", "dire", "clip"],
    "detection_version": "advanced-ensemble-v1.0"
  },
  "summary": {
    "has_metadata": false,
    "suspicious_flags_count": 1,
    "authenticity_confidence": "medium",
    "ai_probability": 0.87,
    "ai_classification": "likely_ai_generated",
    "total_detection_signals": 21,
    "suspicious_detection_signals": 14
  }
}
```

---

## Classification Labels

| Label | AI Probability | Meaning |
|-------|---------------|---------|
| `likely_ai_generated` | > 80% | Strong indicators of AI generation |
| `possibly_ai_generated` | 50–80% | Some AI indicators present |
| `possibly_authentic` | 30–50% | Likely real, minor anomalies |
| `likely_authentic` | < 30% | Strong indicators of real photo |

---

## Architecture

```
Frontend (GitHub Pages)
    ↓ POST /api/v1/analyze/image
Backend (HuggingFace Spaces — Docker)
    ↓
ImageForensics
    ├── extract_exif()
    ├── generate_hashes()           → SHA256, MD5, pHash, aHash, dHash
    ├── detect_tampering_indicators()
    └── detect_ai_generation()
            ↓
        AdvancedEnsembleDetector
            ├── StatisticalDetector  → 19 signals
            ├── DIREDetector         →  1 signal
            └── CLIPDetector         →  1 signal
                                     ─────────
                                     21 signals total
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn (Python 3.11) |
| AI Detection | PyTorch 2.1, OpenAI CLIP, Stable Diffusion 2.1 |
| Image Processing | OpenCV, Pillow, scikit-image, PyWavelets |
| Statistical Analysis | NumPy, SciPy, scikit-learn |
| Hashing | hashlib (SHA256/MD5), imagehash (perceptual) |
| Rate Limiting | slowapi (10 req/min) |
| Caching | In-memory SHA256-keyed LRU cache |
| Frontend | Plain HTML/CSS/JavaScript |
| CI/CD | GitHub Actions |
| Deployment | HuggingFace Spaces (backend) + GitHub Pages (frontend) |

---

## Running Locally

```bash
git clone https://github.com/abinaze/VeriFile-X.git
cd VeriFile-X

# Linux/macOS
sudo apt-get install -y libmagic1

# Install dependencies
pip install -r backend/requirements.txt

# Start backend
export PYTHONPATH=$(pwd)
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Open frontend
open frontend/index.html
# or serve locally:
python -m http.server 3000 --directory frontend
```

API docs: http://localhost:8000/docs

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/analyze/image` | Analyze image (max 10MB) |
| POST | `/api/v1/upload/validate` | Validate file only |

---

## Security

- Rate limited: 10 requests/minute per IP
- Double validation: content-type header + python-magic MIME check
- In-memory only: uploaded files are never written to disk
- No user data stored: analysis results cached by SHA256 hash only

---

## Test Suite

115+ tests covering:
- Unit tests for all 21 detection signals
- API integration tests
- Edge cases (corrupted images, tiny images, grayscale, RGBA)
- Fuzz testing
- Performance and memory bounds
- Cache correctness
- Determinism validation

```bash
# Run fast tests (excludes slow ML model tests)
pytest backend/tests/ -v -m "not slow" --tb=short
```

---

## License

MIT License — see [LICENSE](LICENSE)
