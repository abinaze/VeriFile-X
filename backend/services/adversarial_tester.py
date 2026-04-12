"""
Adversarial Robustness Testing for AI Detection Pipeline.

Tests the detection system against 8 classes of adversarial attacks
that an adversary might use to fool the detector into misclassifying
AI-generated images as authentic.

Attack types tested:
  1. JPEG re-compression      - Repeatedly re-encode to wash out AI artifacts
  2. Gaussian noise injection  - Add noise to disrupt frequency signatures
  3. Mild blurring             - Smooth out GAN/diffusion artifacts
  4. Pixel shuffle             - Random local pixel permutation
  5. Color jitter              - Brightness/contrast/saturation shifts
  6. Downscale-upscale         - Resolution reduction then upsampling
  7. Random cropping           - Remove border regions
  8. Histogram equalization    - Normalize intensity distribution

For each attack the system measures:
  - Score delta: how much the AI probability changed
  - Verdict flip: whether the classification changed
  - Robustness score: 1.0 = completely robust, 0.0 = completely fooled

Target robustness: score delta < 0.20 for mild attacks (intensity 0.3)
"""
import logging
import numpy as np
from typing import Dict, Any, List
from io import BytesIO

logger = logging.getLogger(__name__)

_ATTACK_CONFIGS = {
    "jpeg_recompression": {
        "description": "Repeated JPEG re-encoding to wash out AI frequency artifacts",
        "intensities": [0.3, 0.6, 1.0],
    },
    "gaussian_noise": {
        "description": "Gaussian noise injection to disrupt spectral signatures",
        "intensities": [0.3, 0.6, 1.0],
    },
    "gaussian_blur": {
        "description": "Mild blurring to smooth GAN/diffusion artifacts",
        "intensities": [0.3, 0.6, 1.0],
    },
    "color_jitter": {
        "description": "Brightness and contrast shifts",
        "intensities": [0.3, 0.6, 1.0],
    },
    "downscale_upscale": {
        "description": "Resolution reduction then upsampling (information loss)",
        "intensities": [0.3, 0.6, 1.0],
    },
    "random_crop": {
        "description": "Remove border regions to strip metadata artifacts",
        "intensities": [0.3, 0.6, 1.0],
    },
    "histogram_equalization": {
        "description": "Normalize intensity distribution to reduce compression signals",
        "intensities": [0.3, 0.6, 1.0],
    },
    "pixel_shuffle": {
        "description": "Local pixel permutation to disrupt spatial patterns",
        "intensities": [0.3, 0.6, 1.0],
    },
}


def _apply_jpeg_recompression(img_array: np.ndarray, intensity: float) -> np.ndarray:
    from PIL import Image
    quality = int(95 - intensity * 50)  # 95 -> 45
    rounds  = max(1, int(intensity * 5))
    buf = BytesIO()
    pil = Image.fromarray(img_array.astype(np.uint8), "RGB")
    for _ in range(rounds):
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
    return np.array(pil)


def _apply_gaussian_noise(img_array: np.ndarray, intensity: float) -> np.ndarray:
    sigma = intensity * 25.0
    noise = np.random.default_rng(42).normal(0, sigma, img_array.shape)
    return np.clip(img_array.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def _apply_gaussian_blur(img_array: np.ndarray, intensity: float) -> np.ndarray:
    import cv2
    radius = max(1, int(intensity * 7))
    ksize  = radius * 2 + 1
    return cv2.GaussianBlur(img_array, (ksize, ksize), 0)


def _apply_color_jitter(img_array: np.ndarray, intensity: float) -> np.ndarray:
    from PIL import Image, ImageEnhance
    pil = Image.fromarray(img_array.astype(np.uint8), "RGB")
    for enhancer_cls, factor in [
        (ImageEnhance.Brightness, 1.0 + intensity * 0.4),
        (ImageEnhance.Contrast,   1.0 + intensity * 0.3),
        (ImageEnhance.Color,      1.0 + intensity * 0.2),
    ]:
        pil = enhancer_cls(pil).enhance(factor)
    return np.array(pil)


def _apply_downscale_upscale(img_array: np.ndarray, intensity: float) -> np.ndarray:
    import cv2
    h, w   = img_array.shape[:2]
    scale  = 1.0 - intensity * 0.5
    new_h  = max(32, int(h * scale))
    new_w  = max(32, int(w * scale))
    small  = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _apply_random_crop(img_array: np.ndarray, intensity: float) -> np.ndarray:
    import cv2
    h, w  = img_array.shape[:2]
    crop  = int(min(h, w) * intensity * 0.15)
    if crop < 1:
        return img_array
    cropped = img_array[crop:h-crop, crop:w-crop]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)


def _apply_histogram_equalization(img_array: np.ndarray, intensity: float) -> np.ndarray:
    import cv2
    if intensity < 0.3:
        return img_array
    yuv    = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    # Blend with original based on intensity
    alpha  = min(1.0, intensity)
    return np.clip(
        (1 - alpha) * img_array.astype(np.float64) + alpha * result.astype(np.float64),
        0, 255
    ).astype(np.uint8)


def _apply_pixel_shuffle(img_array: np.ndarray, intensity: float) -> np.ndarray:
    rng       = np.random.default_rng(42)
    result    = img_array.copy()
    h, w      = img_array.shape[:2]
    block     = max(4, int(min(h, w) * 0.05))
    n_blocks  = int(intensity * 20)
    for _ in range(n_blocks):
        y = rng.integers(0, h - block)
        x = rng.integers(0, w - block)
        patch = result[y:y+block, x:x+block].reshape(-1, 3)
        idx   = rng.permutation(len(patch))
        result[y:y+block, x:x+block] = patch[idx].reshape(block, block, 3)
    return result


_ATTACK_FNS = {
    "jpeg_recompression":    _apply_jpeg_recompression,
    "gaussian_noise":        _apply_gaussian_noise,
    "gaussian_blur":         _apply_gaussian_blur,
    "color_jitter":          _apply_color_jitter,
    "downscale_upscale":     _apply_downscale_upscale,
    "random_crop":           _apply_random_crop,
    "histogram_equalization": _apply_histogram_equalization,
    "pixel_shuffle":         _apply_pixel_shuffle,
}

_CLASS_ORDER = [
    "likely_authentic", "possibly_authentic",
    "possibly_ai_generated", "likely_ai_generated"
]


def _to_bytes(arr: np.ndarray) -> bytes:
    from PIL import Image
    buf = BytesIO()
    Image.fromarray(arr.astype(np.uint8), "RGB").save(buf, format="PNG")
    return buf.getvalue()


def run_robustness_test(
    image_bytes: bytes,
    filename: str = "unknown",
    attacks: List[str] = None,
) -> Dict[str, Any]:
    """
    Run adversarial robustness test suite against the detection pipeline.

    Args:
        image_bytes: Original image bytes
        filename:    Image filename for logging
        attacks:     List of attack names to run (default: all 8)

    Returns:
        Dict with per-attack results and overall robustness score
    """
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector

    if attacks is None:
        attacks = list(_ATTACK_CONFIGS.keys())

    # Get baseline detection
    try:
        det_base   = AdvancedEnsembleDetector(image_bytes, filename)
        base       = det_base.detect()
        det_base.cleanup()
        base_score = base["ai_probability"]
        base_class = base["classification"]
    except Exception as e:
        logger.error(f"Baseline detection failed: {e}")
        return {
            "error":            str(e),
            "robustness_score": 0.0,
            "attacks_tested":   0,
        }

    # Load image array
    try:
        from PIL import Image
        pil      = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_arr  = np.array(pil)
    except Exception as e:
        return {"error": f"Image decode failed: {e}", "robustness_score": 0.0}

    attack_results = {}
    total_robustness = []

    for attack_name in attacks:
        if attack_name not in _ATTACK_FNS:
            continue

        attack_fn     = _ATTACK_FNS[attack_name]
        config        = _ATTACK_CONFIGS[attack_name]
        intensity_results = []

        for intensity in config["intensities"]:
            try:
                attacked_arr   = attack_fn(img_arr.copy(), intensity)
                attacked_bytes = _to_bytes(attacked_arr)

                det_atk = AdvancedEnsembleDetector(attacked_bytes, f"{filename}_attacked")
                result  = det_atk.detect()
                det_atk.cleanup()

                atk_score  = result["ai_probability"]
                atk_class  = result["classification"]
                delta      = abs(atk_score - base_score)
                flipped    = (atk_class != base_class)
                rob_score  = max(0.0, 1.0 - delta * 2)

                intensity_results.append({
                    "intensity":        round(intensity, 2),
                    "attacked_score":   round(atk_score, 4),
                    "score_delta":      round(delta, 4),
                    "verdict_flipped":  flipped,
                    "robustness_score": round(rob_score, 4),
                })
                total_robustness.append(rob_score)

            except Exception as e:
                logger.warning(f"Attack {attack_name}@{intensity} failed: {e}")
                intensity_results.append({
                    "intensity":       round(intensity, 2),
                    "error":           str(e),
                    "robustness_score": 1.0,
                })

        attack_results[attack_name] = {
            "description":    config["description"],
            "results":        intensity_results,
            "mean_robustness": round(
                float(np.mean([r["robustness_score"] for r in intensity_results])), 4
            ),
        }

    overall = round(float(np.mean(total_robustness)), 4) if total_robustness else 0.0

    # Classify robustness level
    if overall >= 0.80:
        robustness_level = "high"
        summary_note     = "Detection pipeline is robust against common adversarial attacks."
    elif overall >= 0.60:
        robustness_level = "medium"
        summary_note     = "Detection pipeline shows moderate robustness. Consider adversarial training."
    else:
        robustness_level = "low"
        summary_note     = "Detection pipeline is vulnerable to adversarial attacks. Adversarial training recommended."

    logger.info(
        f"Robustness test complete for {filename}: "
        f"overall={overall:.3f} ({robustness_level}), "
        f"{len(attack_results)} attacks tested"
    )

    return {
        "baseline_score":       round(base_score, 4),
        "baseline_class":       base_class,
        "overall_robustness":   overall,
        "robustness_level":     robustness_level,
        "attacks_tested":       len(attack_results),
        "attack_results":       attack_results,
        "summary":              summary_note,
        "recommendation": (
            "Run scripts/adversarial_train.py to improve robustness via "
            "PGD adversarial training on the ensemble model."
            if overall < 0.80 else
            "No immediate action required. Re-test after model updates."
        ),
    }