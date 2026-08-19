# F-17 Profiling — Parallelizing the 30 Signals

## Method

27 of the 30 signals (19-signal statistical bundle + 8 classical forensic detectors +
`image_type_classifier`) are pure numpy/opencv/scipy/scikit-image — no torch dependency — so they
were timed for real: a synthetic 1600×1200 JPEG (smooth gradient + sensor-like noise, quality 88),
5 runs each, mean/min/max recorded. The 3 torch-dependent signals (DIRE, CLIP, own-embedding)
couldn't be executed in the profiling session's sandbox (no working torch install there) — their
relative cost was estimated by reading the model architecture directly instead.

## Real measured numbers (mean of 5 runs, synthetic 1600×1200 JPEG)

| Signal | Mean | Notes |
|---|---|---|
| Statistical bundle (19 signals, 1 call) | 2116.5ms | Internally: BasicSignals 1396.7ms, UltraSignals 264.5ms, AdvancedSignals 232.2ms, CovarianceSignals 149.8ms |
| JPEG ghost | 1026.1ms | |
| DCT frequency | 758.6ms | |
| Noiseprint | 557.7ms | |
| Noise map | 481.6ms | |
| CFA | 154.8ms | |
| Image type classifier | 126.4ms | Gates prnu/ela/metadata — must resolve before those three, independent of everything else |
| PRNU | 92.0ms | |
| ELA | 75.2ms | |
| Metadata | 0.1ms | |

**Sum if sequential: ~5.4s. Slowest single signal (the parallel floor for this group): ~2.1s
(statistical bundle).**

Within the statistical bundle itself, F-16 already decomposed it into 4 independent components
sharing only a read-only `ImageContext` — `BasicSignals` alone is ~66% of the bundle's own total
(1.4s of 2.1s), so there's a second, smaller layer of parallelization available inside the bundle
if the outer-level win alone isn't enough later. Not implemented in this pass — flagged for
whoever picks up further F-17 work.

## DIRE / CLIP / own-embedding — structural estimate, not measured

Read directly from `dire_detector.py`, `clip_detector.py`, `own_embedding_detector.py`:

- **DIRE**: loads a full Stable Diffusion 2.1 pipeline, runs **20 sequential UNet forward passes**
  (~860M params) at 512×512, plus a VAE encode/decode. Almost certainly the dominant cost of the
  whole pipeline on CPU — likely an order of magnitude past the 2.1s statistical bundle.
- **CLIP**: one forward pass, ViT-B/32 (~88M params). Fast relative to DIRE.
- **own-embedding**: one forward pass, EfficientNet-B0 (~5M params). Fastest of the three.

To get a real DIRE number, run this in an environment with working torch (this repo's own dev
environment, not necessarily the profiling sandbox):

```python
import time, statistics
from backend.services.dire_detector import DIREDetector

with open("some_real_test_image.jpg", "rb") as f:
    image_bytes = f.read()

times = []
d = DIREDetector()
d.detect(image_bytes, "warmup.jpg")  # first call also pays model-load cost — exclude it
for _ in range(3):
    t0 = time.perf_counter()
    d.detect(image_bytes, "test.jpg")
    times.append(time.perf_counter() - t0)
print(f"DIRE mean: {statistics.mean(times):.2f}s  (min {min(times):.2f}s, max {max(times):.2f}s)")
```

## The scheduler-sharing risk — RESOLVED

DIRE's `DDIMScheduler` was loaded once and cached process-wide (`ModelCache`, F-7) —
`cached_model['scheduler']` — with every `DIREDetector` instance sharing that *same object*. Both
`_add_noise()` and the denoising loop call mutating methods on it (`scheduler.add_noise()`,
`scheduler.set_timesteps()`, `scheduler.step()`). `denoise_steps` is hardcoded to `20`, so a
`set_timesteps(20)` race between two concurrent DIRE calls likely produced the same values either
way — probably not silently wrong in practice. But that was incidental, not a guarantee, and it was
exactly the risk the original F-17 planning flagged as "a plausible interaction risk nobody has
tested yet."

**Fixed** (follow-up commit, after this document was first written): `DIREDetector._load_model()`
now clones a fresh scheduler from the cached one's config on every cache hit
(`type(cached_scheduler).from_config(cached_scheduler.config)`) instead of sharing the cached
object directly. `from_config()` does no disk/network I/O — just Python object construction from a
config dict — so this is negligible overhead per request. `self.pipe` (the actual UNet/VAE weights)
stays shared: frozen, eval-mode forward passes don't mutate shared state, so sharing that part
remains safe. Verified directly: two `DIREDetector` instances that both hit the cache now get two
distinct scheduler objects (`first.scheduler is not second.scheduler`); confirmed this is a real
fix, not just plausible, by reverting it and watching the new regression test fail with exactly the
shared-object assertion error, then pass again once restored.

**This does not by itself mean DIRE should join the parallel signal pool.** The remaining open
question is the one this document's "DIRE / CLIP / own-embedding — structural estimate, not
measured" section above already raised: there's still no real DIRE timing number, so it's not known
whether DIRE dominates total latency so heavily that parallelizing it alongside anything else
provides only a marginal win, or whether there's a genuine, substantial gain available. Get a real
number (the script above) before deciding whether extending the thread pool to DIRE/CLIP/
own-embedding is worth the added complexity — the thread-safety blocker is gone, but the
latency-value question isn't answered yet.

## What was actually shipped this round (partial F-17)

The 9 mutually-independent, non-torch signal calls — the statistical bundle + 8 classical
forensic detectors — now run in a `ThreadPoolExecutor` inside `AdvancedEnsembleDetector.detect()`,
instead of one after another. Every module involved was grepped for module-level caches/globals
before this change; none exist (all are pure functions over their own local data), so there's no
DIRE-scheduler-style shared-state risk in this half of the change.

**A real trap found and fixed during verification**: the profiling/development sandbox turned out
to have exactly 1 CPU core. On 1 core, 9 threads contending for that core measured **~1.5x SLOWER**
than plain sequential — pure context-switch overhead, zero real parallelism, not a hypothetical
concern. Fixed by capping `max_workers = min(len(_signal_tasks), max(1, os.cpu_count() or 1))`, so
the pool degrades to one-thread-at-a-time (matching sequential performance) on a constrained host
instead of regressing, and scales up to real concurrency on whatever's actually available.

The project's actual production target (Hugging Face Spaces CPU Basic) is confirmed 2 vCPU — real,
though bounded (not the full 9-way ideal), benefit is expected there. DIRE/CLIP/own-embedding
remain sequential, tracked as a separate follow-up gated on the scheduler-sharing question above
and a real DIRE number from the script in this doc.

## Verification performed

- Ran the real, patched `AdvancedEnsembleDetector.detect()` (torch import stubbed just enough to
  satisfy `dire_detector.py`/`clip_detector.py`'s type annotations at import time; `dire_detector`/
  `clip_detector`/`own_detector` swapped for instant fakes so timing measures only the 9-signal
  pool this change touches) — 3 repeated runs produced identical `ai_probability` and identical
  30-signal sets, confirming no race condition/data corruption from the thread pool.
- Directly verified the `max_workers` capping logic against 3 scenarios: `cpu_count()=1` → 1
  worker, `cpu_count()=64` → 9 workers (capped at task count, not core count), `cpu_count()=None`
  → 1 worker (not passed through as `None`, which `ThreadPoolExecutor` would otherwise interpret
  as "pick a default based on core count," silently reintroducing the oversubscription bug this
  fix exists to prevent).
- Full `backend/` still `py_compile`-clean after the change.
- Could **not** measure real multi-core speedup in the profiling sandbox (1 core, by definition
  can't demonstrate parallelism). Recommend running the timing harness below on this repo's own
  environment (the 2-vCPU HF Space, or local dev hardware) to confirm the real-world win:

```python
import time
from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector

with open("some_real_test_image.jpg", "rb") as f:
    image_bytes = f.read()

t0 = time.perf_counter()
AdvancedEnsembleDetector(image_bytes, "test.jpg").detect()
print(f"Full detect(): {time.perf_counter() - t0:.2f}s")
```
