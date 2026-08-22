# Setting Up VeriFile-X From Scratch

This is a teaching guide, not a quick-reference — it explains *why* each step exists, not just the
command to run. If you just want the fastest path to a running API, the [README](README.md)'s
"Getting Started" section is shorter. Come here when you want to understand the whole system well
enough to rebuild it, extend it, or train it properly yourself.

## Read this first — the honest version

Before anything else: **the live demo and the model files this repo ships with are a
proof-of-concept, not a validated product.** Two specific things you should know going in, because
they explain a lot of what follows:

1. **The bundled "AI-generated" reference images aren't from real AI generators.**
   `scripts/generate_ai_samples.py` — one of the scripts this guide walks through — creates 500
   *synthetic placeholder* images (smooth gradients, symmetric patterns, blurred noise) that
   approximate some AI-image characteristics, because the project didn't have paid access to
   Midjourney/DALL-E/Stable Diffusion generation APIs when it needed reference images fast. The
   script says so in its own docstring. Any model trained only on those placeholders is learning
   "smooth gradient vs. real photo," not "AI-generated vs. real photo" — a much easier, much less
   useful distinction.
2. **The full dataset this project is designed for is ~200,000 images** (see
   `data/DATASETS.md`); what's actually been trained on, in the checked-in reference files, is far
   smaller. Fewer, less diverse examples means the learned components (the fine-tuned classifier
   and the CLIP/embedding centroids) generalize worse to images unlike whatever they saw — which is
   exactly why the live demo can be confidently wrong on real-world images. This isn't a bug to fix
   by tweaking code; it's a direct, mechanical consequence of dataset size and quality, the same way
   it would be for anyone's model.

None of this means the system doesn't work — the 27 classical, non-learned signals (statistical,
PRNU, ELA, DCT, JPEG ghost, noise map, noiseprint, CFA, metadata) don't need training data at all
and are exactly as reliable running from this repo as running from a 200,000-image version. It
means the 3 learned signals (DIRE's reference behavior aside, mainly CLIP-centroid and
OwnEmbedding) are only as good as what you train them on — and this repo's own bundled reference
data is intentionally a stand-in, not the real thing. **If you're setting this up to actually rely
on, budget real time and either a real generation API or a real downloaded dataset for Part 5
below — don't skip straight to "it works" on the placeholder data and assume the numbers mean
anything.**

---

## What you'll end up with

A FastAPI backend serving a 30-signal image-forensics API, a static frontend that talks to it, and
(if you do Parts 5–6) your own trained reference data instead of the placeholder set. Read
[`ARCHITECTURE.md`](docs/ARCHITECTURE.md) alongside this guide for what each piece actually does
once it's running.

## Prerequisites

| Requirement | Why |
|---|---|
| Python 3.11 | Pinned version — the project hasn't been tested on 3.12+ or 3.10- |
| ~6GB free disk | DIRE's Stable Diffusion 2.1 pipeline alone is ~4-5GB; CLIP and the fine-tuned embedding model are much smaller |
| A CPU is enough | Everything here runs CPU-only by design (this project's production target is a 2-vCPU Hugging Face Space); a GPU speeds up training in Part 6 but isn't required |
| Git + Git LFS | The `.pkl`/`.pt` files in `data/reference/` are Git LFS objects — a plain ZIP download only gives you pointer stubs, not the real files (harmless; you'll either train your own in Part 6 or `git lfs pull` if you have real ones) |
| ~30-60 min for Part 1-4, hours to days for Part 5-6 | Getting a running system is fast; building a dataset large enough for the learned signals to mean something is the actual time cost, exactly like any ML project |

---

## Part 1 — Environment

```bash
git clone https://github.com/abinaze/VeriFile-X.git
cd VeriFile-X

python3.11 -m venv venv
source venv/bin/activate        # Windows Git Bash: source venv/Scripts/activate

cd backend
pip install -r requirements-linux.txt --extra-index-url https://download.pytorch.org/whl/cpu
# macOS/Windows: use requirements.txt or requirements-windows.txt instead — see the comment
# at the top of each file for the right --extra-index-url / --index-url, since torch's CPU
# wheels are hosted differently per platform.
```

**Why the separate PyTorch index URL:** without it, `pip` resolves the default CUDA-enabled torch
build from PyPI, which is several gigabytes and pulls in CUDA/cuDNN shared libraries you don't need
for a CPU-only deployment. This one flag is the difference between a ~200MB install and a
multi-gigabyte one.

**Verify the install before moving on:**
```bash
python3 -c "import fastapi, torch, cv2, numpy, scipy; print('core imports OK')"
```
If this fails, fix it here — every later step assumes it passes.

---

## Part 2 — Run it with zero trained models first

Don't jump straight to training. Get the untrained system running first, so you have a known-good
baseline to compare against once you do train something.

```bash
# still inside backend/
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` — you should see the full Swagger UI for every endpoint. Try
`POST /api/v1/analyze/image` with any image file and the demo API key
(`vfx_public_demo_2026` — set `PUBLIC_DEMO_KEY=vfx_public_demo_2026` as an environment variable
first, or generate your own key via `POST /api/v1/keys/create`).

**What you should see with zero real trained models:** a complete report with all 30 signals
present — the 27 non-learned ones will produce real, meaningful scores; DIRE/CLIP/OwnEmbedding will
either run against whatever placeholder reference data shipped in `data/reference/` (if you have
the real LFS files) or degrade gracefully to a neutral, low-confidence result (if you only have LFS
pointer stubs — check `backend/services/*_detector.py` for how each one handles a missing
reference file; none of them crash the whole request). **This is the correct baseline** — if
`ai_probability` looks suspiciously confident here, something's misconfigured, because an
untrained/placeholder system shouldn't be confident.

If this doesn't come up cleanly, stop and fix it before going further — Parts 5–6 assume a working
baseline.

---

## Part 3 — What "trained" actually means for this system, concretely

Three files under `data/reference/` are the actual learned artifacts:

| File | What it is | Built by |
|---|---|---|
| `clip_database.pkl` | Centroid (average) CLIP embedding for "real" images and for "AI" images, computed from your labeled dataset | `scripts/build_clip_database.py` |
| `own_embedding_model.pt` | Fine-tuned EfficientNet-B0 weights — a real, trained neural network classifier | `scripts/train_embedding.py` |
| `own_centroids.pkl` | Centroid embeddings from that fine-tuned model, used for a centroid-distance signal separate from the classifier's own output | `scripts/build_centroids.py` |

Two more files are trained on top of *those*, using the full ensemble's output rather than raw
images:

| File | What it is | Built by |
|---|---|---|
| `ensemble_xgb.pkl` | An XGBoost meta-model that learns to combine all 30 signals, as an optional override of the static weighted-average combiner | `scripts/train_ensemble.py` |
| (Platt calibration params, stored in `platt_calibrator.py`'s own data file) | Recalibrates raw ensemble scores against known-labeled outcomes so "0.8" means roughly "80% confident," not just "high" | `scripts/fit_platt.py` |

Everything in Part 5–6 below is building or rebuilding these five files. **The 27 non-learned
signals need none of this** — they're pure signal-processing code, not trained models, and work
identically regardless of whether you ever run Part 5.

---

## Part 4 — The fast path (what this repo ships with, clearly labeled)

This reproduces what's actually in the repo's reference files today — useful for understanding the
pipeline end-to-end quickly, **not** for producing a trustworthy classifier. Budget under an hour.

```bash
cd ..    # repo root

# 1. Generate 500 synthetic "AI-like" placeholder images (see the warning at the top of
#    this guide — these are NOT real generator output).
python scripts/generate_ai_samples.py

# 2. Get some real photos to pair them with. download_real.py pulls directly from a public
#    dataset's URL:
python scripts/datasets/download_real.py --dataset coco --limit 500

# 3. Index everything you now have into data/manifest.csv (assigns 80/10/10 train/val/test splits):
python scripts/datasets/index_manual.py --source coco --label real
python scripts/datasets/index_manual.py --source synthetic_ai --label ai

# 4. Build the CLIP centroid database from the manifest:
python scripts/build_clip_database.py --max-per-class 500

# 5. Fine-tune the embedding classifier (5 epochs is fast but not enough for real accuracy --
#    it's enough to produce a working .pt file to test the pipeline with):
python scripts/train_embedding.py --epochs 5 --batch 32

# 6. Build centroids from that freshly-trained model:
python scripts/build_centroids.py

# 7. Re-run the baseline analysis from Part 2 and compare -- ai_probability should now move
#    in response to real vs. synthetic-pattern images, since the centroids/classifier have
#    something (even if crude) to compare against.
```

If you did this, you now have a **working demonstration of the pipeline**, not a working
classifier. The rest of this guide is what closes that gap.

---

## Part 5 — Building a dataset that's actually large enough to matter

`data/DATASETS.md` documents the target: roughly 200,000 balanced images (100,000 real, 100,000
AI-generated) from named, licensed sources, split 80/10/10 train/val/test. This is the part that
takes real time and either money (a generation API) or patience (downloading and licensing-checking
public datasets) — there's no shortcut here, and anyone telling you there is one is selling
something.

**Real photos** — `scripts/datasets/download_real.py --dataset <name>` already knows how to pull
several of the sources in `DATASETS.md` (COCO, DIV2K, and others — check `--help` for the current
list). For sources it doesn't automate (RAISE, FFHQ, Unsplash Lite), download manually per that
file's URLs, then run:
```bash
python scripts/datasets/index_manual.py --source <name> --label real
```

**AI-generated images** — two real options, in order of effort:
1. `scripts/datasets/download_ai.py --dataset cifake` (and similar) pulls existing labeled
   AI-image datasets from Kaggle — needs a free Kaggle API token (`~/.kaggle/kaggle.json`, the
   script's docstring has the exact steps). This gets you real generator output (Stable Diffusion,
   GANs, depending on the dataset) at essentially no cost beyond download time.
2. If you want images from a *specific* generator this project doesn't have a dataset source for,
   generate them yourself via a real API (Stable Diffusion, DALL-E, Midjourney) and run
   `index_manual.py --label ai` against the output folder, exactly like a manually-downloaded real
   dataset.

Either way, **replace the `synthetic_ai` placeholder entries in your manifest with real ones** —
don't just add real AI images alongside the synthetic ones and call it done; the synthetic set
actively teaches the wrong distinction if left mixed in at any meaningful proportion.

**Quality rules** (`DATASETS.md` again): real photos need genuine camera EXIF or a verified
raw-photo source, no editing-software fingerprint, minimum 256×256; AI images need a verified
generator label and no camera EXIF. `index_manual.py` checks basic dimensions/EXIF presence
automatically; the generator-label and source-verification parts are on you to track honestly, since
a mislabeled training example is worse than a missing one.

---

## Part 6 — Training each component, in order

Once your manifest reflects a real, properly-sized, properly-labeled dataset:

### 6a. CLIP reference database
```bash
python scripts/build_clip_database.py --model ViT-B/32 --max-per-class 10000 --split train
```
`--max-per-class 0` uses everything in the training split. `--model ViT-L/14` trades slower
embedding for a larger, more accurate CLIP backbone if you have the compute for it.

### 6b. Fine-tune the embedding classifier
```bash
python scripts/train_embedding.py --epochs 20 --batch 32 --lr 3e-4
```
Watch validation accuracy, not just training accuracy — with a real, diverse dataset this should
climb well past what 5 quick epochs on placeholder data ever could. `--freeze-backbone` trains only
the final classification layer (faster, less prone to overfitting on a smaller dataset);
omit it once you have enough data to fine-tune the whole network.

### 6c. Rebuild centroids from the newly-trained model
```bash
python scripts/build_centroids.py
```

### 6d. Train the XGBoost meta-model
```bash
python scripts/train_ensemble.py --test-size 0.15 --early-stop 20
```
This trains on the *combined output* of all 30 signals against your labeled data, learning
non-linear interactions the static weighted average can't capture. Requires the CLIP/embedding
steps above to already be in place, since it needs their signal outputs as input features.

### 6e. Calibrate probabilities
```bash
python scripts/fit_platt.py --max-iter 500 --lr 0.01
```
`--dry-run` shows what would change without writing the calibration file — useful for a quick sanity
check before committing to a fit.

### Evaluate what you actually built
```bash
python scripts/evaluate_model.py --threshold 0.5
```
This is the step that tells you, honestly, whether the above was worth it — precision/recall/F1 on
a held-out test split, not just a vibe. Compare this against a run using the original placeholder
data to see the real difference dataset quality makes.

---

## Part 7 — Verify before you trust it

```bash
python scripts/production_check.py --strict
python scripts/generate_model_hashes.py    # populates data/reference/known_hashes.json for real,
                                            # instead of the empty {} this repo ships with
cd backend && pytest tests/ -v -m "not slow" --timeout=60
pytest tests/ -v -m "slow" --timeout=300   # the tests that actually load DIRE/CLIP for real
```

`production_check.py --strict` is the project's own automated sanity gate — it exists specifically
to catch "looks configured but isn't" mistakes before you deploy. Don't skip it.

---

## Part 8 — Frontend

The frontend is a single static HTML file with no build step:
```bash
cd frontend
python3 -m http.server 8080
```
Open `http://localhost:8080` — it auto-detects `localhost`/`127.0.0.1` and points itself at
`http://localhost:8000` for the API, so it'll talk to your local backend from Part 2 without any
config changes. To point it at a deployed backend instead, edit the `API_URL` constant near the top
of `index.html`'s script block, and make sure `PUBLIC_DEMO_KEY` there matches your backend's
`PUBLIC_DEMO_KEY` setting exactly (see [`DEPLOYMENT.md`](DEPLOYMENT.md) — this exact mismatch has
caused real, hard-to-diagnose "silently broken public demo" issues before).

---

## Part 9 — Deploying

Covered in full in [`DEPLOYMENT.md`](DEPLOYMENT.md) — Hugging Face Spaces (this project's actual
production target) and Render are both documented there, including the environment-variable
gotchas that have genuinely broken this deployment before (`PUBLIC_DEMO_KEY` in particular has to be
set directly in the platform's own settings UI, not just in a committed config file).

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `torch` install grabs several GB | Missing the `--extra-index-url https://download.pytorch.org/whl/cpu` flag from Part 1 |
| DIRE/CLIP/own-embedding all return neutral, low-confidence results | Expected if `data/reference/*.pkl`/`*.pt` are still Git LFS pointer stubs, not real files — either `git lfs pull` (if real trained files exist upstream) or work through Parts 4-6 to build your own |
| "Invalid or inactive API key" on the public demo | `PUBLIC_DEMO_KEY` mismatch between frontend and backend — see Part 8 |
| `ai_probability` looks suspiciously confident on an untrained/placeholder system | Something's misconfigured — an undertrained system being *confidently* wrong is a red flag, not a good sign; re-check Part 2's baseline |
| Training accuracy climbs but validation accuracy doesn't | Classic overfitting on too small/undiverse a dataset — this is the exact failure mode the honest-limitations warning at the top of this guide is about |
| A test in `test_clip_database.py`/`test_dire_detector.py`/etc. fails in CI or a sandboxed environment | Often environmental (no network to the model CDN, no GPU) rather than a real regression — see [`docs/TESTING_STATUS.md`](docs/TESTING_STATUS.md)'s "Known environmental gaps" section before assuming it's a bug |

## Where to go next

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how a request actually flows through the system
- [`data/DATASETS.md`](data/DATASETS.md) — full dataset source list and quality rules
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — Hugging Face Spaces / Render deployment
- [`SECURITY.md`](SECURITY.md) — the security model and how to report an issue
- [`README.md`](README.md#accuracy-validation-and-honest-limitations) — the full, specific accuracy
  and validation caveats, including the known data-leakage issue in the one training run this
  project has actually completed
