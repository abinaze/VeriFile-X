# VeriFile-X

**Privacy-Preserving AI-Powered Digital Forensics Platform**

[![CI Status](https://github.com/abinaze/VeriFile-X/actions/workflows/ci.yml/badge.svg)](https://github.com/abinaze/VeriFile-X/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A digital forensics platform that analyzes images to detect AI-generated content, tampering, and authenticity issues using statistical analysis and metadata extraction. **Zero file storage** - all processing happens in-memory.

## 🎯 What This Does

VeriFile-X takes an uploaded image and provides:
- **EXIF Metadata Analysis** - Camera info, GPS coordinates, timestamps, software traces
- **Cryptographic Hashing** - SHA-256, MD5, perceptual hashes for integrity verification
- **AI Detection** - Statistical analysis to identify AI-generated images (70-80% accuracy)
- **Tampering Detection** - Identify editing software traces and missing metadata
- **Forensic Report** - Comprehensive JSON output with confidence scores

**Privacy-First:** No files stored, no databases, all analysis in-memory.

## 🚀 Quick Demo
```bash
# Start the server
uvicorn backend.main:app --reload

# Analyze an image
curl -X POST "http://localhost:8000/api/v1/analyze/image" \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "file_info": {
    "filename": "photo.jpg",
    "format": "JPEG",
    "size": [1920, 1080]
  },
  "ai_detection": {
    "ai_probability": 0.23,
    "classification": "likely_authentic",
    "confidence": "high"
  },
  "hashes": {
    "sha256": "a3f2...",
    "perceptual_hash": "f8e4..."
  },
  "summary": {
    "has_metadata": true,
    "authenticity_confidence": "high"
  }
}
```

## ✨ Features

### Implemented
- ✅ **File Validation** - MIME type detection, size limits, malicious file rejection
- ✅ **EXIF Metadata Extraction** - Camera make/model, GPS, timestamps, software
- ✅ **Hash Generation** - SHA-256, MD5, perceptual hashing
- ✅ **AI Detection** - Statistical analysis (noise, frequency, JPEG artifacts, color)
- ✅ **Tampering Detection** - Software traces, missing metadata indicators
- ✅ **Rate Limiting** - 10 requests/minute per IP (DoS protection)
- ✅ **Caching** - SHA-256 keyed in-memory cache (47x speedup on duplicates)
- ✅ **CI/CD** - GitHub Actions automated testing
- ✅ **API Documentation** - Auto-generated Swagger UI

### Planned
- 🔄 Frontend web UI
- 🔄 Video forensics
- 🔄 Batch processing
- 🔄 PDF document analysis

## 🏗️ Architecture
```
Client → Rate Limiter → Validation → Cache Check → Forensics Pipeline → Response
                                          ↓
                               (EXIF, Hashes, AI Detection, Tampering)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI (async ASGI) |
| **Validation** | python-magic (MIME detection) |
| **Image Processing** | Pillow, OpenCV |
| **Analysis** | NumPy, SciPy (FFT), scikit-learn |
| **Caching** | In-memory (SHA-256 keyed) |
| **Testing** | pytest (31 tests, 100% critical path) |
| **CI/CD** | GitHub Actions |

## 📦 Installation

### Prerequisites
- Python 3.11+
- pip

### Setup
```bash
# Clone repository
git clone https://github.com/abinaze/VeriFile-X.git
cd VeriFile-X

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Create .env file
cp backend/.env.example backend/.env
```

### Run Server
```bash
# From project root
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Server runs at `http://localhost:8000`
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## 🧪 Testing
```bash
# Run all tests
export PYTHONPATH="${PWD}:${PYTHONPATH}"
pytest backend/tests/ -v

# With coverage
pytest backend/tests/ --cov=backend --cov-report=html

# Specific test file
pytest backend/tests/test_ai_detector.py -v
```

**Test Coverage:** 31 tests covering validators, forensics, AI detection, API endpoints, caching.

## 📚 API Endpoints

### `POST /api/v1/upload/validate`
Quick file validation (type + size check).

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/upload/validate" \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "valid": true,
  "mime_type": "image/jpeg",
  "size_mb": 2.4,
  "filename": "image.jpg"
}
```

### `POST /api/v1/analyze/image`
Full forensic analysis (rate limited: 10/min).

**Features:**
- EXIF metadata extraction
- Hash generation (SHA-256, MD5, perceptual)
- AI detection via 4 statistical signals
- Tampering indicator analysis
- Confidence scoring

**Rate Limit:** 10 requests/minute per IP

See `/docs` for interactive API documentation.

## 🔬 AI Detection Methodology

**Approach:** Statistical signal analysis (no heavy ML models required)

**Detection Signals:**

1. **Noise Pattern Analysis** (Laplacian operator)
   - Real photos: Natural sensor noise (Gaussian distribution)
   - AI images: Artificial uniformity

2. **Frequency Domain** (2D FFT)
   - Real photos: Natural spectral decay
   - AI images: Abnormal frequency signatures

3. **JPEG Artifacts** (DCT block analysis)
   - Real photos: Authentic compression patterns
   - AI images: Over-smoothed or missing artifacts

4. **Color Distribution** (HSV entropy)
   - Real photos: Natural color variance
   - AI images: Oversaturation or uniform distribution

**Accuracy:** 70-80% on typical AI-generated images

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for mathematical details.

## 🔐 Security & Privacy

### Security Layers
1. **Rate Limiting** - 10 requests/min per IP
2. **Multi-layer Validation** - Content-type, MIME, size checks
3. **Input Sanitization** - python-magic file signature verification
4. **Memory Safety** - In-memory only processing
5. **Type Safety** - Pydantic models for all I/O

### Privacy Guarantees
- ✅ **Zero file storage** - Files never touch disk
- ✅ **No databases** - No persistent data storage
- ✅ **In-memory only** - All processing in RAM
- ✅ **No PII logging** - File content never logged
- ✅ **GDPR compliant** - No personal data retention

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Analysis time (cache miss)** | ~235ms |
| **Analysis time (cache hit)** | ~5ms |
| **Cache speedup** | 47x |
| **Max file size** | 10MB |
| **Rate limit** | 10 req/min per IP |

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔒 Security

Report vulnerabilities to: abinazebinoy@gmail.com

See [SECURITY.md](SECURITY.md) for responsible disclosure policy.

## 👤 Author

**Abinaze Binoy**  
📧 abinazebinoy@gmail.com  
🔗 [GitHub](https://github.com/abinaze)

## 🙏 Acknowledgments

- FastAPI for excellent async framework
- scikit-image & OpenCV for vision tools
- Pillow for EXIF handling
- The open-source community

---

**Status:** Production-ready backend, frontend in development  
**Version:** 1.0.0  
**Last Updated:** February 2026
