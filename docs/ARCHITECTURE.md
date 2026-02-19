# VeriFile-X Architecture

## System Overview

VeriFile-X is a privacy-preserving digital forensics platform that analyzes images for authenticity using statistical analysis and metadata extraction.
```
┌─────────────────────────────────────────────────────────────┐
│                        Client                               │
│                   (Browser / curl)                          │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTPS
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Rate Limiter (10 req/min)               │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           API Routes (/api/v1/...)                   │   │
│  │  • /upload/validate  - File validation               │   │
│  │  • /analyze/image    - Forensic analysis             │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Validation Layer                        │   │
│  │  • MIME type check (python-magic)                    │   │
│  │  • Size limit (10MB)                                 │   │
│  │  • Content-type header validation                    │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           SHA-256 Hash Cache                         │   │
│  │  • Check for duplicate (cache hit)                   │   │
│  │  • Return cached result if found                     │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Forensics Analysis Pipeline                  │   │
│  │                                                      │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  1. Metadata Extraction (EXIF, GPS)            │  │   │
│  │  │     - Camera make/model                        │  │   │
│  │  │     - GPS coordinates                          │  │   │
│  │  │     - Software used                            │  │   │
│  │  │     - Timestamps                               │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  2. Hash Generation                            │  │   │
│  │  │     - SHA-256 (cryptographic)                  │  │   │
│  │  │     - MD5 (legacy)                             │  │   │
│  │  │     - Perceptual hash (similarity)             │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  3. AI Detection (Statistical)                 │  │   │
│  │  │     - Noise pattern analysis (Laplacian)       │  │   │
│  │  │     - Frequency domain (2D FFT)                │  │   │
│  │  │     - JPEG artifacts (DCT blocks)              │  │   │
│  │  │     - Color distribution (HSV entropy)         │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  4. Tampering Detection                        │  │   │
│  │  │     - Missing EXIF indicators                  │  │   │
│  │  │     - Software manipulation traces             │  │   │
│  │  │     - AI generation signatures                 │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │                                                      │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Response Generation                        │   │
│  │  • Compile forensic report (JSON)                    │   │
│  │  • Cache result for duplicates                       │   │
│  │  • Return to client                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. API Layer (`backend/api/`)
- **Purpose:** HTTP endpoint routing and request handling
- **Technology:** FastAPI (async ASGI)
- **Security:** Rate limiting (slowapi), CORS, input validation
- **Routes:**
  - `POST /api/v1/upload/validate` - Fast file validation
  - `POST /api/v1/analyze/image` - Full forensic analysis

### 2. Validation Layer (`backend/utils/`)
- **Purpose:** Multi-layer file security validation
- **Components:**
  - Content-type header check (fast fail)
  - python-magic MIME verification (reads file signature)
  - Size limit enforcement (10MB max)
  - Malicious file rejection

### 3. Cache System (`backend/core/cache.py`)
- **Purpose:** Performance optimization via deduplication
- **Implementation:** In-memory SHA-256 keyed cache
- **Strategy:** LRU eviction, 60min TTL, 500 entry limit
- **Privacy:** Stores analysis results only, never file bytes

### 4. Forensics Engine (`backend/services/`)

#### Image Forensics (`image_forensics.py`)
- **EXIF Extraction:** Pillow + ExifTags parsing
- **GPS Decoding:** DMS to decimal degrees conversion
- **Hash Generation:** SHA-256, MD5, perceptual (imagehash)
- **Tampering Detection:** Software traces, missing metadata

#### AI Detector (`ai_detector.py`)
- **Approach:** Statistical analysis (no heavy ML models)
- **Signals:**
  1. **Noise Analysis:** Laplacian operator + local variance
     - Metric: `consistency = σ_local / μ_local`
     - Real photos: Higher variance diversity
  2. **Frequency Domain:** 2D FFT spectral analysis
     - Metric: `ratio = LowFreq / HighFreq`
     - AI images: Abnormal spectral signatures
  3. **JPEG Artifacts:** DCT block boundary analysis
     - Metric: Blockiness + edge density
     - AI images: Over-smoothed or missing artifacts
  4. **Color Distribution:** HSV histogram entropy
     - Metric: `H(X) = -Σ p(x)log p(x)`
     - AI images: Lower entropy, oversaturation

### 5. Configuration (`backend/core/config.py`)
- **Pydantic Settings:** Type-safe env var management
- **Environment-based:** DEBUG, CORS_ORIGINS, file size limits
- **Security:** No secrets in code, .env for local dev

### 6. Testing (`backend/tests/`)
- **Framework:** pytest + pytest-asyncio
- **Coverage:** 31 tests across all modules
- **Strategy:** Unit tests per component + integration tests for API

## Data Flow

### Typical Request Flow

1. **Client uploads image** → POST /api/v1/analyze/image
2. **Rate limiter** checks IP (10 req/min limit)
3. **Validation** checks content-type, size, MIME
4. **Cache lookup** via SHA-256 hash
   - **Cache hit:** Return cached result (instant)
   - **Cache miss:** Continue to analysis
5. **Forensics pipeline:**
   - Extract EXIF metadata
   - Generate hashes (SHA-256, perceptual)
   - Run AI detection (4 statistical signals)
   - Detect tampering indicators
6. **Report generation** (JSON)
7. **Cache storage** for future duplicates
8. **Response** to client with complete analysis

## Security Architecture

### Defense Layers

1. **Rate Limiting:** 10 requests/minute per IP
2. **Input Validation:**
   - Content-type header check
   - python-magic MIME type verification
   - 10MB size limit
3. **Memory Safety:**
   - All processing in-memory
   - No disk writes (privacy-first)
   - File handles closed in `finally` blocks
4. **Type Safety:** Pydantic models for all I/O
5. **Logging:** Structured logs without PII

### Privacy Guarantees

- **Zero file storage:** Files never touch disk
- **In-memory only:** Bytes processed in RAM
- **No PII logging:** File content never logged
- **Cache privacy:** Stores results only, not file data
- **Auto-cleanup:** Cache clears on restart

## Performance Characteristics

### Timing Breakdown (100x100 image)

| Operation | Time | Cacheable |
|-----------|------|-----------|
| File validation | ~5ms | No |
| EXIF extraction | ~10ms | Yes |
| Hash generation | ~15ms | Yes |
| AI detection | ~200ms | Yes |
| Tampering check | ~5ms | Yes |
| **Total (cache miss)** | **~235ms** | - |
| **Total (cache hit)** | **~5ms** | - |

### Scalability Considerations

- **Bottleneck:** AI detection (CPU-intensive FFT)
- **Cache benefit:** 47x speedup on duplicates
- **Rate limiting:** Prevents DoS on CPU-heavy ops
- **Async I/O:** Non-blocking file reads
- **Horizontal scaling:** Stateless design (cache per instance)

## Technology Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.109.0 | Async web framework |
| Pydantic | 2.5.3 | Data validation |
| python-magic | 0.4.27 | MIME type detection |
| Pillow | 10.2.0 | Image processing + EXIF |
| OpenCV | 4.9.0 | Computer vision operations |
| NumPy | 1.26.3 | Numerical computing |
| SciPy | 1.11.4 | Scientific computing (FFT) |
| imagehash | 4.3.1 | Perceptual hashing |
| slowapi | 0.1.9 | Rate limiting |

### Development Tools

- **Testing:** pytest, pytest-asyncio, httpx
- **Linting:** (recommended: ruff, black)
- **CI/CD:** GitHub Actions
- **Python:** 3.11+ (for performance)

## Design Decisions

### Why Statistical AI Detection (Not Deep Learning)?

**Pros:**
- ✅ No heavy model downloads (TensorFlow/PyTorch)
- ✅ Fast inference (<1 second)
- ✅ Interpretable results (signal breakdown)
- ✅ Works offline
- ✅ Lower memory footprint

**Cons:**
- ⚠️ ~70-80% accuracy vs ~90%+ with CNNs
- ⚠️ Vulnerable to adversarial attacks

**Justification:** For a portfolio/learning project, statistical approach demonstrates understanding of signal processing, computer vision fundamentals, and engineering tradeoffs without requiring GPU infrastructure.

### Why In-Memory Only (No Database)?

**Pros:**
- ✅ True privacy (nothing persisted)
- ✅ Simpler deployment (no DB management)
- ✅ Faster (no I/O overhead)
- ✅ GDPR/privacy compliant by design

**Cons:**
- ⚠️ No historical analysis
- ⚠️ Cache lost on restart
- ⚠️ No user accounts/sessions

**Justification:** Privacy-first design is the core value proposition. For a forensics tool, users may not want their files tracked.

### Why FastAPI (Not Flask/Django)?

- **Async support:** Non-blocking I/O for file uploads
- **Auto documentation:** OpenAPI/Swagger UI
- **Type safety:** Pydantic integration
- **Performance:** Faster than Flask
- **Modern:** Python 3.11+ features

## Future Enhancements

### Short-term (Next Iterations)
1. Frontend UI (React/Vue)
2. Video forensics support
3. Document analysis (PDF tampering)
4. Batch processing endpoint

### Long-term (Production)
1. CNN-based AI detection (higher accuracy)
2. Redis cache (persistent across restarts)
3. PostgreSQL for audit logs (optional)
4. Kubernetes deployment
5. WebSocket real-time progress
6. Fine-tuned models on custom dataset

## Development Setup

See main README for full setup instructions.

## Testing
```bash
# Run all tests
pytest backend/tests/ -v

# Run with coverage
pytest backend/tests/ --cov=backend --cov-report=html

# Run specific module
pytest backend/tests/test_ai_detector.py -v
```

## Deployment

Currently designed for single-instance deployment (Render, Railway, fly.io). For production scale, consider:
- Load balancer (Nginx)
- Redis for shared cache
- Separate worker processes for CPU-heavy operations
- CDN for static assets

---

**Last Updated:** February 2026  
**Version:** 1.0.0  
**Author:** Abinaze Binoy
