# VeriFile-X — Phase Roadmap

All branches follow the naming convention `feature/phase-N-name` or `fix/description` and merge into `main`.

---

## Status Key

| Symbol | Meaning |
|--------|---------|
| DONE | Merged to main, tagged |
| FIXED | Bug fix committed |
| PLANNED | Design complete, ready to build |
| FUTURE | Scoped, design in progress |

---

## Completed Phases

### Phase 1 — Project Foundation
Core FastAPI application, configuration via pydantic-settings, structured logging, CORS, lifespan startup, health endpoint, Dockerfile, GitHub Actions CI.

### Phase 2 — Image Validation and Upload
MIME type validation, magic-byte checking via `python-magic`, size limits, upload endpoint, extension filtering, error messages.

### Phase 3 — Statistical Detection Engine
Noise floor analysis, DCT frequency artifact detection, Benford's Law on pixel values, KL divergence against natural image spectral model. 12+ signals in `AdvancedAIDetector` and `UltraAdvancedDetector`.

### Phase 4 — Deep Learning Detection (DIRE + CLIP)
DIRE diffusion reconstruction error (ICCV 2023 paper). CLIP universal fake detection (UnivFD CVPR 2023). Own embedding detector with custom reference database and Siamese-style centroid comparison.

### Phase 5 — Advanced Ensemble
Weighted ensemble across all detectors. Normalised weights summing to exactly 1.0. XGBoost meta-model override when trained model is available. Platt-style calibration stub. Explainability: which signals contributed most to the verdict.

### Phase 6 — ELA and Metadata Forensics
Error Level Analysis with 2-sigma concentration threshold. Deep EXIF/metadata forensics using the public Pillow `getexif()` API. Format awareness for lossless inputs.

### Phase 7 — Heatmap and Attribution
Patch-based Grad-CAM manipulation localization heatmap. Generator attribution classifier: Stable Diffusion, DALL-E 3, SDXL, Midjourney, StyleGAN.

### Phase 8 — Platform Detection and C2PA
Social media platform fingerprint detection (Instagram, Twitter, Facebook, LinkedIn, WhatsApp). C2PA content credential verification.

### Phase 9 — Forensic Report and Export
Full forensic report structure with all 26 signal scores. PDF, JSON, and CSV export via `report_exporter.py`. SHA-256 integrity hash per report. Evidence ID generation.

### Phase 10 — API Key Management and RBAC
API key creation, revocation, and verification. Roles: admin, analyst, viewer. SHA-256 key hashing — raw keys never stored.

### Phase 11 — Batch Processing
Parallel batch analysis up to 10 images. Aggregate batch verdict: high_risk, mixed, or likely_authentic. Per-image results in a single response.

### Phase 12 — Adversarial Robustness Testing
Tests whether detection holds under JPEG re-compression, Gaussian blur, additive noise, and histogram equalization.

### Phase 13 — Caching and Performance
Thread-safe `ForensicsCache` with TTL expiry. SHA-256 pre-computed hash accepted to avoid duplicate hashing. Latency tracking via `perf_counter`.

### Phase 14 — Security Hardening
Sliding window rate limiter via slowapi. IP SHA-256 hashing in logs. Security headers: HSTS, CSP, X-Frame-Options, Permissions-Policy. Input validation and injection detection.

### Phase 15 — Audit Log and Provenance
Append-only SHA-256 hash-chained audit log. Concurrent write lock. Timestamped rotation on size. `settings.VERSION` in every record.

### Phase 16 — Case Management
Investigation case system with JSONL persistence. Evidence attachment. Case search, status management, and summary generation. Append-only case store — last snapshot per case_id wins.

### Phase 17 — Monitoring and Observability
Metrics collector. System metrics endpoint. Admin-protected metrics reset (`X-Admin-Key` header).

### Phase 18 — SSE Streaming
Server-Sent Events endpoint for real-time per-signal streaming. 26 signals streamed as they complete. Rate-limited: 5/minute.

---

### Fix 11 — `fix(batch): fix uncertain_count negative values`
**File:** `backend/services/batch_processor.py`
`"authentic"` substring matching was non-exclusive — a classification like `"possibly_authentic"` could be counted in both `real_count` and potentially affect `uncertain_count` in unexpected ways. Fixed with exclusive classification: `authentic` only if `ai_generated` not in the string.

### Fix 12 — `fix(ensemble): resolve XGB path via __file__; log when own_weight=0`
**File:** `backend/services/advanced_ensemble_detector.py`
`Path("data/reference/ensemble_xgb.pkl")` resolved relative to CWD — the model was never found when the server was started from any directory other than the project root. Fixed with `Path(__file__).parent.parent.parent / "data" / ...`. Added a `logger.debug()` when `own_weight` is excluded so the silent renormalization is visible in logs.

### Fix 13 — `security(admin): validate X-Admin-Key against hash, not just length`
**File:** `backend/main.py`
The admin key check only verified `len(key) >= 16` — any 16-character string passed. Added SHA-256 comparison against `ADMIN_KEY_HASH` environment variable using `secrets.compare_digest()` to prevent timing attacks.

### Fix 14 — `fix(hsts): send Strict-Transport-Security only over HTTPS`
**File:** `backend/main.py`
HSTS was unconditionally added to every response including local HTTP development traffic. Per RFC 6797, HSTS sent over HTTP is ignored by browsers but is technically invalid and confusing. Now only sent when `request.url.scheme == "https"` or `X-Forwarded-Proto: https` is set.

### Fix 15 — `chore: bump VERSION to 7.2.0`
**File:** `backend/core/config.py`
Version incremented to reflect the fix batch.

---

## Planned Phases

### Phase 19 — Webhook System

**Summary:** Outbound webhook delivery so downstream systems receive analysis results without polling.

**Files:**
- `backend/services/webhook_manager.py` — register, sign (HMAC-SHA256), retry (3×: 5s / 30s / 120s), delivery log
- `backend/api/routes/webhooks.py` — register, list, delete, test, delivery log endpoints
- Wire `fire_webhooks()` into analyze endpoint on completion

**Version:** 7.2.0 → 7.3.0

---

### Phase 20 — JPEG Ghost and Noise Map Detectors

**Summary:** Two new forensic signals based on double-JPEG compression detection and residual noise map analysis.

**Key algorithms:**
- **JPEG Ghost:** Re-compress at each quality level 51–99; energy minimum at original quality reveals ghost artifacts
- **Noise Map:** `noise = original - gaussian_filtered(original)`; analyze frequency spectrum, spatial variance, and regional consistency

**Files:**
- `backend/services/jpeg_ghost_detector.py`
- `backend/services/noise_map_detector.py`
- Add both to ensemble with calibrated weights; renormalize

**Version:** 7.3.0 → 7.4.0

---

### Phase 21 — Noiseprint Learned Camera Fingerprint

**Summary:** Upgrade the PRNU detector with a learning-based camera fingerprint. Noiseprint trains a CNN to suppress scene content and enhance model-specific residuals — consistently outperforms classical PRNU on forgery localization.

**Key algorithm:**
```
residual(patch) = patch - CNN_denoiser(patch)
forgery score  = 1 - cosine_similarity(patch_residual, image_residual)
```

**Files:**
- `backend/services/noiseprint_detector.py`
- Add signal to ensemble and SSE stream

**Version:** 7.4.0 → 7.5.0

---

### Phase 22 — CFA Artifact Analysis

**Summary:** Color Filter Array inter-pixel correlation analysis. AI-generated images have no Bayer sensor pattern — CFA analysis detects the absence of this expected correlation.

**Key algorithm:**
```
skip0_std = std(green[:, :-1] - green[:, 1:])
skip1_std = std(green[:, :-2] - green[:, 2:])
cfa_ratio = skip0_std / skip1_std
# Real cameras: cfa_ratio ~ 0.7–1.0
# AI images: cfa_ratio close to 1.0 (no pattern)
```

**Version:** 7.5.0 → 7.6.0

---

### Phase 23 — Signed Reports and Chain of Custody

**Summary:** Cryptographically signed PDF reports and a public verification endpoint for third-party report validation.

**What it builds:**
- Ed25519 report signing
- `GET /api/v1/verify/{evidence_id}` — public endpoint, no API key required
- Chain-of-custody JSON block embedded in every export
- QR code in PDF linking to verification endpoint

**Version:** 7.6.0 → 7.7.0

---

### Phase 24 — MCMC Probabilistic Authenticity Engine

**Summary:** Replace the single point-estimate with a probability distribution using Markov Chain Monte Carlo sampling over the signal space.

**Output (new fields):**
```json
{
  "probability_distribution": {
    "point_estimate": 0.87,
    "interval_90": [0.74, 0.96],
    "interval_50": [0.81, 0.92],
    "std": 0.06,
    "certainty": "high"
  }
}
```

**Why:** Two images scoring 0.87 can have very different evidential quality — one with all signals agreeing (certain) and one with conflicting signals (uncertain). MCMC makes this visible.

**Version:** 7.7.0 → 7.8.0

---

### Phase 25 — Platt Scaling Confidence Calibration

**Summary:** Replace the current one-line `calibrate()` stub with a properly fitted Platt scaling layer trained on a labeled holdout set.

**Algorithm:**
```
P(y=1 | f) = sigmoid(A * f + B)
where A, B fitted by maximum likelihood on calibration holdout
```

Wilson score intervals provide 90% confidence bounds.

**Version:** 7.8.0 → 7.9.0

---

### Phase 26 — Stable Evidence IDs

**Summary:** Replace random UUID4 evidence IDs with deterministic UUID5 derived from the file's SHA-256 hash.

```python
evidence_id = uuid.uuid5(uuid.NAMESPACE_URL, file_sha256)
```

Same file always produces the same evidence_id, enabling cross-case deduplication and historical lookup.

**Version:** 7.9.0 → 8.0.0

---

### Phase 27 — Segment-Level AI Detection

**Summary:** Detect partial AI insertion — real background with AI-generated subject composited in.

**What it builds:**
- `POST /api/v1/analyze/segment` — per-tile probability grid at N×N resolution
- Frontend: grid overlay visualization
- Algorithm: 64×64 overlapping tiles, per-tile ELA + DCT + noise consistency score

**Version:** 8.0.0 → 8.1.0

---

### Phase 28 — TIFF and HEIC Format Support

**Summary:** Accept professional camera formats so unmodified originals can be analyzed without lossy conversion.

**Why:** Forcing users to convert TIFF/HEIC to JPEG before upload destroys EXIF data and alters ELA baselines — exactly the evidence the system needs.

**Changes:** Add `image/tiff` and `image/heic` to `ALLOWED_IMAGE_TYPES`; install `pillow-heif` and register opener at startup.

**Version:** 8.1.0 → 8.2.0

---

### Phase 29 — Nash Equilibrium Adaptive Detection

**Summary:** Signal weight self-adjustment based on analyst feedback. When analysts mark a result as wrong, the signals that were most mislead receive lower weight on similar future inputs.

**New endpoint:** `POST /api/v1/feedback` — accepts `evidence_id` + `true_label` + optional notes

**Version:** 8.2.0 → 8.3.0

---

### Phase 30 — Multi-Scale Forgery Localization

**Summary:** Transformer-based dense self-attention localization, replacing the Grad-CAM heatmap. Captures long-range cross-image inconsistencies that CNNs miss.

**Algorithm:**
- Three scales: full image, 2×2 tiles, 4×4 tiles
- Feature extraction with pretrained ResNet-50
- Self-attention maps at each scale
- Cross-scale fusion → localization mask

**Version:** 8.3.0 → 8.4.0

---

### Phase 31 — Production Hardening and v1.0.0 Launch

**Summary:** Load testing, penetration test, documentation completion, and official v1.0.0 release.

**Checklist:**
- [ ] 100 concurrent user load test, 30-minute sustained
- [ ] All 341+ tests passing at ≥ 80% coverage
- [ ] Penetration test on all API endpoints
- [ ] README, DEPLOYMENT, PHASE_ROADMAP complete
- [ ] All Dependabot alerts resolved or documented
- [ ] `v1.0.0` tagged and GitHub release published

---

## GitHub Issue Template

Use this when opening an issue for a new phase or bug fix:

```markdown
## Phase N — [Feature Name] / Fix: [Bug Description]

**Type:** Feature | Bug | Security | Performance
**Priority:** High | Medium | Low

### Summary
One paragraph describing what this adds or fixes and why it matters.

### Files
**New:**
- `path/to/new_file.py` — purpose

**Modified:**
- `path/to/existing.py` — what changes

### Acceptance Criteria
- [ ] All existing tests still pass
- [ ] New tests written and passing
- [ ] No new pyflakes warnings
- [ ] VERSION bumped in `config.py`
- [ ] Documented

### Version bump
`X.Y.Z` → `X.Y.(Z+1)`

Closes #[issue]
```

---

## Pull Request Template

```markdown
## [Phase N / Fix] — [Short Description]

### What this does
[One paragraph: what changed and why.]

### Files changed
**New:** `path/file.py` — purpose
**Modified:** `path/file.py` — what and why

### Test results
- New tests: [N] in `test_X.py`
- Total passing: [N]
- Coverage: [N]%
- Pyflakes: clean

### How to verify
```bash
pytest backend/tests/test_X.py -v
pytest backend/tests/ --tb=short
```

Closes #[N]
```
