---
title: VeriFile-X API
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

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

- **Statistical Analysis (19 signals)** — FFT, DCT, wavelet, covariance, Mahalanobis distance, KL divergence, and more
- **DIRE** — Diffusion Reconstruction Error (ICCV 2023), detects Stable Diffusion / DALL-E 3 / Midjourney
- **CLIP** — Semantic embedding similarity (CVPR 2023), zero-shot generalization

**Ensemble:** `0.40 × Statistical + 0.35 × DIRE + 0.25 × CLIP`

---

## API

`POST /api/v1/analyze/image` — upload an image, get full forensic report
`GET /health` — health check

---

## Tech Stack

FastAPI · PyTorch · OpenAI CLIP · Stable Diffusion · OpenCV · scikit-learn · NumPy
