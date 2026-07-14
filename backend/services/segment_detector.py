"""
Segment-Level AI Detection — Phase 27.

Detects partial AI insertion: real background with an AI-generated
subject composited in. Returns a per-tile probability grid showing
which regions of the image are most likely AI-generated.

Algorithm
---------
1. Divide image into overlapping 64×64 tiles (stride = 32 px).
2. For each tile compute three fast signals:
   - ELA score: re-compress at Q=92, measure mean absolute error
   - DCT score: high-frequency coefficient ratio (AC vs DC energy)
   - Noise score: local std of Laplacian residual
3. Normalise each signal per-image (min-max across all tiles).
4. Weighted combination: 0.45×ELA + 0.35×DCT + 0.20×noise.
5. Smooth the grid with a 3×3 Gaussian kernel to remove salt-and-pepper.
6. Return the grid as a 2-D list of floats plus a summary.

Output format
-------------
{
  "grid":        [[float, ...], ...],   # rows × cols probabilities
  "grid_rows":   int,
  "grid_cols":   int,
  "tile_size":   64,
  "stride":      32,
  "max_score":   float,
  "mean_score":  float,
  "hot_tiles":   int,    # tiles with score > 0.6
  "coverage":    float,  # fraction of tiles above threshold
}
"""

import io
import numpy as np
from typing import Any, Dict, List
from PIL import Image

from backend.core.logger import setup_logger

logger = setup_logger(__name__)

_TILE     = 64
_STRIDE   = 32
_ELA_Q    = 92
_HOT_THR  = 0.6


# ── Per-tile signal functions ─────────────────────────────────────────────────

def _ela_score(tile_arr: np.ndarray) -> float:
    """Re-compress tile at Q=92 and measure mean absolute error."""
    img = Image.fromarray(tile_arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_ELA_Q)
    buf.seek(0)
    recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
    return float(np.mean(np.abs(tile_arr.astype(np.float32) - recomp)))


def _dct_score(tile_gray: np.ndarray) -> float:
    """High-frequency DCT coefficient energy ratio."""
    from scipy.fft import dct
    d = dct(dct(tile_gray.astype(np.float64), axis=0, norm="ortho"),
            axis=1, norm="ortho")
    total = float(np.sum(d ** 2)) + 1e-10
    # High-frequency: bottom-right quadrant
    hf = float(np.sum(d[_TILE // 2:, _TILE // 2:] ** 2))
    return hf / total


def _noise_score(tile_gray: np.ndarray) -> float:
    """Local noise level via Laplacian residual standard deviation."""
    from scipy.ndimage import laplace
    lap = laplace(tile_gray.astype(np.float64))
    return float(np.std(lap))


# ── Main function ─────────────────────────────────────────────────────────────

def detect_segments(image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """
    Run segment-level AI detection and return a probability grid.
    """
    try:
        img   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr   = np.array(img, dtype=np.uint8)
        gray  = np.array(img.convert("L"), dtype=np.float64)
        h, w  = arr.shape[:2]

        if h < _TILE or w < _TILE:
            return _fallback("Image too small for segment analysis")

        tiles_ela   : List[float] = []
        tiles_dct   : List[float] = []
        tiles_noise : List[float] = []
        positions   : List[tuple]  = []   # (row_idx, col_idx)

        rows = list(range(0, h - _TILE + 1, _STRIDE))
        cols = list(range(0, w - _TILE + 1, _STRIDE))

        for ri, r in enumerate(rows):
            for ci, c in enumerate(cols):
                tile_rgb  = arr[r:r + _TILE, c:c + _TILE]
                tile_gray = gray[r:r + _TILE, c:c + _TILE]
                tiles_ela.append(_ela_score(tile_rgb))
                tiles_dct.append(_dct_score(tile_gray))
                tiles_noise.append(_noise_score(tile_gray))
                positions.append((ri, ci))

        def _norm(vals: List[float]) -> np.ndarray:
            """
            BUG FIX: this previously did pure per-image min-max scaling —
            (a - min) / (max - min) — which mathematically GUARANTEES the
            single highest-raw-value tile in ANY image maps to exactly 1.0
            and the lowest to exactly 0.0, regardless of the image's
            absolute characteristics. Since hot_tiles/coverage are then
            thresholded at a fixed 0.6 on this per-image-relative scale,
            EVERY image — authentic or not — tends to show some "hot"
            tiles purely from ordinary variation (a sharp subject against
            soft background, texture next to a flat area, a shadow). A
            portrait with natural depth-of-field blur was a plausible
            false-positive under the old design; it could not distinguish
            "this tile looks different because it's an inserted AI patch"
            from "this tile looks different because that's what any
            ordinary photo's foreground vs. background looks like."

            Fix: z-score relative to the image's OWN mean/std, then squash
            through a sigmoid centered on a fixed anomaly threshold
            (in std-devs above the mean). A tile only scores "hot" if it
            is a genuine statistical outlier relative to the rest of the
            image — not merely "the single most different tile in this
            particular photo." A uniform image (std≈0) now correctly
            produces a low, non-alarming score for every tile, rather
            than guaranteeing one tile hits the maximum.
            """
            a = np.array(vals, dtype=np.float64)
            mean, std = a.mean(), a.std()
            if std < 1e-10:
                # No meaningful variation between tiles at all — nothing
                # can be a statistical outlier. Return a low, non-alarming
                # score rather than the old behavior (guaranteed 0.5, or
                # under true min-max, a guaranteed 0.0/1.0 split).
                return np.full_like(a, 0.3)
            z = (a - mean) / std
            _ANOMALY_Z = 1.5   # tiles beyond ~1.5 std above the image's own mean
            _STEEPNESS = 2.0
            return 1.0 / (1.0 + np.exp(-_STEEPNESS * (z - _ANOMALY_Z)))

        ela_n   = _norm(tiles_ela)
        dct_n   = _norm(tiles_dct)
        noise_n = _norm(tiles_noise)

        combined = 0.45 * ela_n + 0.35 * dct_n + 0.20 * noise_n

        # Build 2-D grid
        n_rows = len(rows)
        n_cols = len(cols)
        grid_flat = np.zeros((n_rows, n_cols), dtype=np.float64)
        for idx, (ri, ci) in enumerate(positions):
            grid_flat[ri, ci] = combined[idx]

        # Smooth with 3×3 Gaussian
        from scipy.ndimage import gaussian_filter
        grid_smooth = gaussian_filter(grid_flat, sigma=1.0)
        grid_smooth = np.clip(grid_smooth, 0.0, 1.0)

        grid_list  = [[round(float(v), 4) for v in row] for row in grid_smooth]
        max_score  = float(grid_smooth.max())
        mean_score = float(grid_smooth.mean())
        hot_tiles  = int(np.sum(grid_smooth > _HOT_THR))
        coverage   = round(hot_tiles / grid_smooth.size, 4)

        logger.info(
            "Segment detection: file=%s grid=%dx%d hot=%d coverage=%.3f",
            filename, n_rows, n_cols, hot_tiles, coverage,
        )

        return {
            "grid":       grid_list,
            "grid_rows":  n_rows,
            "grid_cols":  n_cols,
            "tile_size":  _TILE,
            "stride":     _STRIDE,
            "max_score":  round(max_score, 4),
            "mean_score": round(mean_score, 4),
            "hot_tiles":  hot_tiles,
            "coverage":   coverage,
        }

    except Exception as exc:
        logger.warning("Segment detection failed for %s: %s", filename, exc, exc_info=True)
        return _fallback(f"Segment detection unavailable: {exc}")


def _fallback(reason: str) -> Dict[str, Any]:
    return {
        "grid":       [],
        "grid_rows":  0,
        "grid_cols":  0,
        "tile_size":  _TILE,
        "stride":     _STRIDE,
        "max_score":  0.0,
        "mean_score": 0.0,
        "hot_tiles":  0,
        "coverage":   0.0,
        "error":      reason,
    }
