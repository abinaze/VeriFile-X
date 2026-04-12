"""
DCT Frequency Domain Artifact Detection.

GAN-generated images and some diffusion models leave characteristic
artifacts in the Discrete Cosine Transform (DCT) frequency domain.
These artifacts are invisible to the human eye but mathematically
detectable as spectral peaks or unusual energy distributions.

Key papers:
- "Detecting GAN-Generated Fake Images Using Co-occurrence Matrices" (2019)
- "Leveraging Frequency Analysis for Deep Fake Image Recognition" (ICML 2020)
- "Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks are
   Failing to Reproduce Spectral Distributions" (CVPR 2020)
"""
import numpy as np
from typing import Dict, Any
from PIL import Image
from io import BytesIO
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


def _compute_dct_2d(block: np.ndarray) -> np.ndarray:
    """Compute 2D DCT using scipy if available, else numpy."""
    try:
        from scipy.fft import dctn
        return dctn(block, norm='ortho')
    except ImportError:
        # Fallback: use FFT as approximation
        return np.abs(np.fft.fft2(block))


def detect_dct_artifacts(image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """
    Detect DCT frequency domain artifacts in image.

    Method:
    1. Convert to grayscale, divide into 8x8 blocks (like JPEG)
    2. Compute DCT of each block
    3. Analyze frequency energy distribution across blocks
    4. Check for characteristic GAN/AI spectral peaks
    5. Measure high-frequency energy (AI images are often too smooth)
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("L")
        arr = np.array(img, dtype=np.float64)
        h, w = arr.shape

        if h < 32 or w < 32:
            return {
                "signal_name": "DCT Frequency Artifacts",
                "score": 0.5,
                "confidence": 0.0,
                "explanation": "Image too small for DCT analysis",
                "method": "dct_frequency"
            }

        # === Signal 1: 8x8 block DCT energy distribution ===
        block_size = 8
        dct_blocks = []
        hf_ratios = []  # High-frequency to total energy ratios

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = arr[y:y+block_size, x:x+block_size]
                block = block - block.mean()  # Zero-center

                dct = _compute_dct_2d(block)
                dct_abs = np.abs(dct)

                total_energy = np.sum(dct_abs ** 2) + 1e-10

                # High freq = bottom-right of DCT block
                hf_energy = np.sum(dct_abs[4:, 4:] ** 2)
                hf_ratio = hf_energy / total_energy
                hf_ratios.append(hf_ratio)
                dct_blocks.append(dct_abs)

        if not hf_ratios:
            return {
                "signal_name": "DCT Frequency Artifacts",
                "score": 0.5,
                "confidence": 0.1,
                "explanation": "Insufficient blocks for DCT analysis",
                "method": "dct_frequency"
            }

        mean_hf = float(np.mean(hf_ratios))

        # === Signal 2: Global FFT spectrum analysis ===
        # AI images often have unnaturally smooth spectra
        fft = np.fft.fft2(arr)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        magnitude_log = np.log1p(magnitude)

        # Radial power spectrum
        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.ogrid[:h, :w]
        radius = np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2).astype(int)
        max_r = min(cy, cx)

        radial_power = []
        for r in range(1, max_r):
            mask = radius == r
            if mask.sum() > 0:
                radial_power.append(float(np.mean(magnitude_log[mask])))

        if len(radial_power) > 10:
            # Natural images: power decreases roughly as 1/f
            # AI images: often too uniform or have peaks
            rp = np.array(radial_power)
            rp_diff = np.diff(rp)

            # Measure smoothness of spectral rolloff
            spectral_smoothness = float(np.std(rp_diff))

            # Check for spectral peaks (GAN artifact)
        else:
            spectral_smoothness = 0.5

        # === Signal 3: Checkerboard artifact detection ===
        # Up-convolution in GANs creates checkerboard patterns at 2x, 4x, 8x freq
        fft_magnitude = np.abs(np.fft.fft2(arr))
        h2, w2 = h // 2, w // 2
        # Check for energy at Nyquist-related frequencies
        nyquist_energy = float(np.mean(fft_magnitude[h2-2:h2+2, w2-2:w2+2]))
        total_fft_energy = float(np.mean(fft_magnitude))
        checkerboard_ratio = nyquist_energy / (total_fft_energy + 1e-10)

        # === Compute AI score ===
        # Low high-frequency energy = AI (too smooth)
        if mean_hf < 0.02:
            hf_score = 0.75
        elif mean_hf < 0.05:
            hf_score = 0.55
        elif mean_hf < 0.15:
            hf_score = 0.35
        else:
            hf_score = 0.20

        # Very smooth spectrum (low variation) = AI
        if spectral_smoothness < 0.05:
            smooth_score = 0.70
        elif spectral_smoothness < 0.10:
            smooth_score = 0.50
        else:
            smooth_score = 0.25

        # Checkerboard artifacts = GAN
        if checkerboard_ratio > 3.0:
            checker_score = 0.80
        elif checkerboard_ratio > 1.5:
            checker_score = 0.55
        else:
            checker_score = 0.25

        ai_score = float(np.clip(
            0.45 * hf_score + 0.35 * smooth_score + 0.20 * checker_score,
            0.0, 1.0
        ))

        confidence = min(0.75, 0.35 + (h * w) / (1024 * 1024) * 0.40)

        if mean_hf < 0.03:
            explanation = (
                f"Very low high-frequency DCT energy ({mean_hf:.3f}) — "
                "image is unnaturally smooth, consistent with AI synthesis"
            )
        elif checkerboard_ratio > 2.0:
            explanation = (
                f"Checkerboard frequency artifacts detected "
                f"(ratio={checkerboard_ratio:.2f}) — "
                "typical of GAN up-convolution artifacts"
            )
        else:
            explanation = (
                f"Normal DCT frequency distribution "
                f"(HF={mean_hf:.3f}, smoothness={spectral_smoothness:.3f})"
            )

        logger.info(
            f"DCT analysis: score={ai_score:.3f}, "
            f"hf={mean_hf:.3f}, checker={checkerboard_ratio:.2f}, "
            f"file={filename}"
        )

        return {
            "signal_name": "DCT Frequency Artifacts",
            "score": ai_score,
            "confidence": confidence,
            "explanation": explanation,
            "raw_value": mean_hf,
            "expected_range": "< 0.03 HF energy for AI images",
            "method": "dct_frequency"
        }

    except Exception:
        logger.warning("DCT frequency analysis failed", exc_info=True)
        return {
            "signal_name": "DCT Frequency Artifacts",
            "score": 0.5,
            "confidence": 0.0,
            "explanation": "DCT analysis unavailable.",
            "raw_value": 0.0,
            "method": "dct_frequency"
        }
