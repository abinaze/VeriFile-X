"""
Image Type Classifier — first-stage gate before forensic scoring.

Classifies images into content types so detectors can be applied
selectively. A screenshot should not run through PRNU/ELA/metadata
heuristics designed for camera photos.

Content types:
  photo          - Natural photograph from camera or similar
  screenshot_ui  - Screenshot or UI capture (no camera noise)
  illustration   - Drawing, render, graphic, or artwork
  document       - Scanned document or text-heavy image
  low_info       - Blank, flat, or near-uniform — too little information
  unknown        - Cannot determine type with confidence

Accuracy note:
  Rule-based. ~80% accuracy on clear cases.
  Edge cases (memes, edited photos, AI art) may be misclassified.
"""
import logging
import numpy as np
from typing import Dict, Any
from io import BytesIO

logger = logging.getLogger(__name__)


def classify_image_type(image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """
    Classify the content type of an image.

    Returns:
        Dict with keys:
          image_type      - str label
          confidence      - float 0.0-1.0
          is_photo        - bool (True if likely camera photo)
          run_prnu        - bool (True if PRNU signal is appropriate)
          run_ela         - bool (True if ELA signal is appropriate)
          run_metadata    - bool (True if metadata signal is appropriate)
          explanation     - str reason for classification
    """
    try:
        from PIL import Image
        import cv2

        img      = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        fmt      = img.format or "unknown"
        arr      = np.array(img, dtype=np.float64)
        gray     = np.mean(arr, axis=2)

        h, w     = gray.shape
        pixels   = h * w

        # ── Feature extraction ─────────────────────────────────────────────────
        # 1. Texture / variance
        std_global = float(gray.std())

        # 2. Edge density (Canny)
        import cv2 as _cv2
        gray_u8  = gray.astype(np.uint8)
        edges    = _cv2.Canny(gray_u8, 50, 150)
        edge_density = float(np.count_nonzero(edges) / max(edges.size, 1))

        # 3. Color saturation
        try:
            rgb_flat = arr.reshape(-1, 3) / 255.0
            # Approximate saturation from RGB
            r, g, b  = rgb_flat[:, 0], rgb_flat[:, 1], rgb_flat[:, 2]
            mx       = np.maximum(np.maximum(r, g), b)
            mn       = np.minimum(np.minimum(r, g), b)
            sat      = np.where(mx > 0, (mx - mn) / mx, 0.0)
            mean_sat = float(sat.mean())
        except Exception:
            mean_sat = 0.3

        # 4. JPEG compression history
        is_jpeg = fmt.upper() in ("JPEG", "JPG")

        # 5. Aspect ratio (screenshots often have specific ratios)
        aspect = width / max(height, 1)

        # 6. Color count (illustrations have fewer unique colors)
        sample = img.resize((64, 64)).quantize(64)
        unique_colors = len(set(list(sample.getdata())))

        # ── Classification rules ───────────────────────────────────────────────

        # Low information — blank, flat, nearly uniform
        if std_global < 8.0 or pixels < 4096:
            return _result("low_info", 0.90,
                is_photo=False, run_prnu=False, run_ela=False, run_metadata=False,
                explanation=f"Image has very low variance ({std_global:.1f}) or is too small — insufficient forensic information.")

        # Screenshot / UI — high edge density, low noise, often PNG, common aspect ratios
        is_common_screen_ratio = any(abs(aspect - r) < 0.1 for r in [16/9, 4/3, 16/10, 1.0])
        if (not is_jpeg and edge_density > 0.08 and std_global < 60 and
                unique_colors < 40 and is_common_screen_ratio):
            return _result("screenshot_ui", 0.75,
                is_photo=False, run_prnu=False, run_ela=False, run_metadata=False,
                explanation="High edge density, low noise, non-JPEG format — likely screenshot or UI capture.")

        # Document — very high edge density, low saturation, near-grayscale
        if edge_density > 0.12 and mean_sat < 0.15 and std_global > 30:
            return _result("document", 0.70,
                is_photo=False, run_prnu=False, run_ela=True, run_metadata=False,
                explanation="High edge density, low color saturation — likely scanned document or text image.")

        # Illustration / render — low noise, restricted palette, high saturation areas
        if unique_colors < 30 and std_global > 20 and mean_sat > 0.4:
            return _result("illustration", 0.65,
                is_photo=False, run_prnu=False, run_ela=False, run_metadata=False,
                explanation="Limited color palette and high saturation — likely illustration or render.")

        # Natural photo — JPEG or high variance, moderate edges, natural noise
        if is_jpeg and std_global > 25 and 0.02 < edge_density < 0.15:
            return _result("photo", 0.80,
                is_photo=True, run_prnu=True, run_ela=True, run_metadata=True,
                explanation="JPEG format with natural texture and edge distribution — likely camera photograph.")

        # PNG but photo-like
        if std_global > 40 and mean_sat > 0.15 and edge_density > 0.03:
            return _result("photo", 0.60,
                is_photo=True, run_prnu=True, run_ela=False, run_metadata=True,
                explanation="Natural texture and color distribution — likely photograph (non-JPEG format).")

        # Default unknown
        return _result("unknown", 0.40,
            is_photo=False, run_prnu=True, run_ela=is_jpeg, run_metadata=True,
            explanation="Cannot determine image type with confidence. Running full analysis.")

    except Exception:
        logger.warning("Image type classification failed for %s", filename, exc_info=True)
        return _result("unknown", 0.0,
            is_photo=False, run_prnu=True, run_ela=True, run_metadata=True,
            explanation="Classification failed — running full analysis.")


def _result(image_type: str, confidence: float,
            is_photo: bool, run_prnu: bool, run_ela: bool, run_metadata: bool,
            explanation: str) -> Dict[str, Any]:
    return {
        "image_type":   image_type,
        "confidence":   round(confidence, 3),
        "is_photo":     is_photo,
        "run_prnu":     run_prnu,
        "run_ela":      run_ela,
        "run_metadata": run_metadata,
        "explanation":  explanation,
    }
