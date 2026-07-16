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
Append-only audit log (JSONL). Concurrent write lock. Timestamped rotation on size. `settings.VERSION` in every record. **Correction (July 2026 audit):** earlier drafts of this document and the README described this as "hash-chained" — it is not; there is no previous-entry-hash linking field. It is append-only, not tamper-evident, and is now described accordingly everywhere in the docs.

### Phase 16 — Case Management
Investigation case system with JSONL persistence. Evidence attachment. Case search, status management, and summary generation. Append-only case store — last snapshot per case_id wins.

### Phase 17 — Monitoring and Observability
Metrics collector. System metrics endpoint. Admin-protected metrics reset (`X-Admin-Key` header).

### Phase 18 — SSE Streaming
Server-Sent Events endpoint for real-time per-signal streaming. 26 signals streamed as they complete. Rate-limited: 5/minute.

---

## Phase 19 and Later

### Phase 19 — Webhook System

**Status:** DONE — confirmed shipped, verified against the live codebase during the July 2026 security & accuracy audit (Phase 32 below).


**Summary:** Outbound webhook delivery so downstream systems receive analysis results without polling.

**Files:**
- `backend/services/webhook_manager.py` — register, sign (HMAC-SHA256), retry (3×: 5s / 30s / 120s), delivery log
- `backend/api/routes/webhooks.py` — register, list, delete, test, delivery log endpoints
- Wire `fire_webhooks()` into analyze endpoint on completion

**Version:** 7.2.0 → 7.3.0

---

### Phase 20 — JPEG Ghost and Noise Map Detectors

**Status:** DONE — confirmed shipped, verified against the live codebase during the July 2026 security & accuracy audit (Phase 32 below).


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

**Status:** DONE — confirmed shipped, verified against the live codebase during the July 2026 security & accuracy audit (Phase 32 below).


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

**Status:** DONE — confirmed shipped, verified against the live codebase during the July 2026 security & accuracy audit (Phase 32 below).


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

**Status:** NOT IMPLEMENTED. **Correction (July 2026 audit):** this phase was never built, despite
the README at one point describing Ed25519 signing, a `chain_of_custody` report field, and a
`GET /api/v1/verify/{evidence_id}` endpoint as though they existed. A repository-wide search
confirms there is no `ed25519`/`chain_of_custody`/`verify_url` anywhere in the codebase, and no
`/api/v1/verify/*` router is mounted anywhere. This entry is left here, corrected, rather than
deleted, specifically so this gap has a visible paper trail instead of silently vanishing.

**Summary:** Cryptographically signed PDF reports and a public verification endpoint for third-party report validation. Still genuinely useful; still not built.

**What it would build:**
- Ed25519 report signing
- `GET /api/v1/verify/{evidence_id}` — public endpoint, no API key required
- Chain-of-custody JSON block embedded in every export
- QR code in PDF linking to verification endpoint

**Target version:** to be scheduled — not committed to a specific release yet.

---

### Phase 24 — MCMC Probabilistic Authenticity Engine

**Status:** DONE — confirmed shipped, verified against the live codebase during the July 2026 security & accuracy audit (Phase 32 below).


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

**Status:** DONE — confirmed shipped, verified against the live codebase during the July 2026 security & accuracy audit (Phase 32 below).


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

**Status:** DONE — confirmed shipped, verified against the live codebase during the July 2026 security & accuracy audit (Phase 32 below).


**Summary:** Replace random UUID4 evidence IDs with deterministic UUID5 derived from the file's SHA-256 hash.

```python
evidence_id = uuid.uuid5(uuid.NAMESPACE_URL, file_sha256)
```

Same file always produces the same evidence_id, enabling cross-case deduplication and historical lookup.

**Version:** 7.9.0 → 8.0.0

---

### Phase 27 — Segment-Level AI Detection

**Status:** DONE — confirmed shipped, verified against the live codebase during the July 2026 security & accuracy audit (Phase 32 below).


**Summary:** Detect partial AI insertion — real background with AI-generated subject composited in.

**What it builds:**
- `POST /api/v1/analyze/segment` — per-tile probability grid at N×N resolution
- Frontend: grid overlay visualization
- Algorithm: 64×64 overlapping tiles, per-tile ELA + DCT + noise consistency score

**Version:** 8.0.0 → 8.1.0

---

### Phase 28 — TIFF and HEIC Format Support

**Status:** DONE — confirmed shipped, verified against the live codebase during the July 2026 security & accuracy audit (Phase 32 below).


**Summary:** Accept professional camera formats so unmodified originals can be analyzed without lossy conversion.

**Why:** Forcing users to convert TIFF/HEIC to JPEG before upload destroys EXIF data and alters ELA baselines — exactly the evidence the system needs.

**Changes:** Add `image/tiff` and `image/heic` to `ALLOWED_IMAGE_TYPES`; install `pillow-heif` and register opener at startup.

**Version:** 8.1.0 → 8.2.0

---

### Phase 29 — Analyst Feedback Weight Adaptation

**Correction (July 2026 audit):** originally titled "Nash Equilibrium Adaptive Detection" — the mechanism is a straightforward per-signal weight penalty on incorrect predictions, not a game-theoretic equilibrium computation. Renamed to describe what it actually does.

**Status:** DONE — confirmed shipped, verified against the live codebase during the July 2026 security & accuracy audit (Phase 32 below).


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
- [ ] All 480+ tests passing at ≥ 80% coverage
- [ ] Penetration test on all API endpoints
- [ ] README, DEPLOYMENT, PHASE_ROADMAP complete
- [ ] All Dependabot alerts resolved or documented
- [ ] `v1.0.0` tagged and GitHub release published

---

### Phase 32 — Security, Accuracy, and Code Quality Audit Remediation

**Status:** DONE (merged via `fix/full-audit-security-accuracy-cleanup`, PR #158).

**Summary:** A full line-by-line professional audit of the backend, frontend, and CI configuration
surfaced 14 critical/high findings and roughly 30 further medium/low findings. This phase is the
complete remediation, verified fix-by-fix against the live codebase rather than assumed from a
plan — several early attempts at these fixes silently failed to apply (a Python patch script's
`assert` raised without halting the shell, producing "nothing to commit, working tree clean") and
had to be re-diagnosed and reapplied correctly; see the commit history on the branch above for the
full sequence.

**Security:**
- Fixed stored XSS in the frontend (filename, EXIF values, and tampering-flag text are now
  HTML-escaped before being rendered)
- Added authentication to `analyze.py` and `cases.py` — previously the two most sensitive routers
  in the app had none at all
- Consolidated four independently-drifting inline auth checks (`keys.py`, `webhooks.py`,
  `feedback.py` ×2) into one shared, tested dependency (`backend/core/auth.py`)
- Added SSRF protection to webhook registration (hostname resolution plus a private/loopback/
  link-local/reserved/multicast IP check)
- Fixed an admin-endpoint auth bypass that was live in the project's own real deployment
  configuration (the gate now fails closed by default)
- Added rate limiting to `webhooks.py`, which previously had none on any of its five endpoints

**Accuracy:**
- Fixed `noiseprint_detector.py`'s reference-patch bug, which made the signal mathematically
  incapable of returning anything but a constant, confidently-wrong "authentic" score
- Unified the ensemble's two independently-maintained scoring branches into one confidence-gated,
  feedback-weighted path, so the recommended (DIRE-enabled) deployment no longer misses
  refinements that previously only applied to the fallback path
- Fixed DIRE's noise draw to use a per-image seeded generator instead of the shared global RNG,
  restoring same-image determinism
- Fixed `segment_detector.py`'s per-image min-max normalization, which mathematically guaranteed
  "hot tiles" on every image, authentic or not
- Fixed the GPS plausibility check in metadata forensics, which never applied the hemisphere sign
  and so could not meaningfully fail on any input

**Correctness and documentation honesty:**
- Corrected the PRNU and C2PA signals' documentation and confidence to match what they actually
  check, rather than the more legally/forensically loaded techniques their naming implied
- Enforced `extension_valid`, which was previously computed and then silently discarded
- Fixed two lossless-format ordering bugs (`ela_detector.py`, `jpeg_ghost_detector.py`)
- Removed the `Ed25519`/`chain_of_custody`/"court-ready" and "hash-chained audit log" claims from
  the README, SECURITY.md, and this file (see Phase 15 and Phase 23 corrections above) — these
  were confirmed, by exhaustive repository search, to not exist in the code

**Configuration and dead code:**
- Wired `CACHE_TTL_MINUTES`, `MAX_CACHE_SIZE`, `LOG_LEVEL`, and `RATE_LIMIT_PER_MINUTE` to the code
  paths they were always documented as controlling, but previously did not
- Removed confirmed-dead code: an unused legacy schema file, a superseded path-based RBAC function,
  unused `CacheConfig` fields, two dead detector methods, and the dead internals of the PDF writer
  class (`build()` never touched any of the instance state `__init__` set up)
- Standardized all 18 remaining bare `logging.getLogger(__name__)` call sites onto the project's
  own `setup_logger()`, restoring visibility into logs that were previously silently dropped
  (nothing in the app calls `logging.basicConfig()`, so unconfigured loggers defaulted to level
  WARNING with no handler)
- Gave `OwnEmbeddingDetector` the same shared-model-cache treatment its sibling detectors (DIRE,
  CLIP) already had, and skipped a wasted ImageNet-weight download that was always immediately
  discarded anyway

**Documentation and repository hygiene:**
- Full rewrite of README.md, SECURITY.md, and this file, grounded in what the code actually does
- New non-commercial LICENSE (PolyForm Noncommercial 1.0.0), replacing MIT
- New SVG project logo, replacing a Git-LFS-pointer PNG that did not actually resolve to an image
  in a plain repository download

---

### Phase 33 — Outstanding Follow-Ups From the Phase 32 Audit

**Status:** PLANNED. Tracked here rather than left implicit, so they do not quietly disappear the
way Phase 23 did.

- Re-validate the ensemble/XGBoost meta-model on a larger, resolution-matched dataset with grouped,
  leakage-checked cross-validation (split by source/generation run, not by individual image) — see
  [README: Accuracy, Validation, and Honest Limitations](README.md#accuracy-validation-and-honest-limitations)
- Re-check webhook SSRF safety at delivery time, not only at registration, to defeat DNS rebinding
  between the two
- Add rotation/pruning to `data/api_keys.jsonl`, which currently grows by one line per successful
  authentication with no bound
- Rename the four test files that still carry their original numbered-phase names
  (`test_phase20.py`, `test_phase21_22.py`, `test_phase24_26.py`, `test_phase27_29.py`) to
  content-based names
- Address the pre-existing mypy/flake8 backlog, then remove `continue-on-error: true` from those
  CI steps so real findings can actually block a merge — deliberately sequenced after the backlog
  cleanup, not before, so this doesn't retroactively block unrelated work
- Resolve or document the currently-open `transformers`/`torch` CVEs once upstream ships a stable
  fixed release

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
