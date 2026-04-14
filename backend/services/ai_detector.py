"""
AI-generated image detection service.
Uses statistical analysis and heuristics to detect AI-generated images.

Detection Signals:
1. Noise pattern consistency  - Sensor noise modeling (Laplacian variance)
2. Frequency domain analysis  - FFT spectral fingerprinting
3. JPEG compression artifacts - DCT block boundary detection
4. Color distribution entropy - HSV histogram analysis

Mathematical basis:
- Noise:     Consistency = σ_local / μ_local  (lower = suspicious)
- Frequency: Ratio = LowFreqEnergy / HighFreqEnergy
- Entropy:   H(X) = -Σ p(x)log p(x)

Note: Detection accuracy ~70-80% on older AI models (pre-2024).
Modern generators (DALL-E 3, Midjourney v6, SDXL) often evade statistical detection.
"""
import numpy as np
import cv2
from scipy import fft
from scipy.stats import entropy
from typing import Dict, Any
from PIL import Image
from io import BytesIO

from backend.core.logger import setup_logger

logger = setup_logger(__name__)


class AIDetector:
    """
    AI-generated image detector using statistical analysis.

    Why statistical approach?
    - No heavy model downloads required
    - Fast inference (< 1 second)
    - Interpretable signal breakdown
    - Works fully offline
    
    Limitations:
    - Modern AI generators (2024+) have improved significantly
    - Statistical signals alone achieve ~70-80% accuracy
    - For production: combine with CNN-based detection
    """

    def __init__(self, image_bytes: bytes, filename: str):
        """
        Initialize detector with image data.

        Args:
            image_bytes: Raw image file content
            filename: Original filename for logging

        Raises:
            ValueError: If image is corrupted or unreadable
        """
        self.image_bytes = image_bytes
        self.filename = filename

        # Load via PIL (for metadata-aware loading)
        self.pil_image = Image.open(BytesIO(image_bytes))

        # Load via OpenCV (for numerical analysis)
        self.cv_image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        # Guard against corrupted/unreadable images
        if self.cv_image is None:
            raise ValueError(f"Invalid or corrupted image file: {filename}")

        self.cv_gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

        logger.info(f"Initialized AI detector for {filename} "
                    f"({self.cv_gray.shape[1]}x{self.cv_gray.shape[0]}px)")

    def analyze_noise_patterns(self) -> Dict[str, Any]:
        """
        Analyze noise patterns using Laplacian operator.

        Mathematical basis:
            L(x,y) = ∇²I(x,y)  (second derivative = high freq noise)
            Consistency = σ_local / μ_local

        Real photos:  Noise ~ N(0, σ²) - natural Gaussian sensor noise
        AI images:    Low stochastic variation → lower variance diversity

        Returns:
            Dictionary with noise metrics
        """
        # Laplacian extracts high-frequency noise components
        laplacian = cv2.Laplacian(self.cv_gray, cv2.CV_64F)
        noise_variance = laplacian.var()

        # Local variance analysis (real photos have higher local diversity)
        kernel_size = 5
        img_float = self.cv_gray.astype(float)
        mean_local = cv2.blur(img_float, (kernel_size, kernel_size))
        sqr_mean = cv2.blur(img_float ** 2, (kernel_size, kernel_size))
        local_variance = sqr_mean - mean_local ** 2

        local_var_mean = local_variance.mean()
        local_var_std = local_variance.std()

        # Consistency ratio: lower = more uniform = more suspicious
        noise_consistency = local_var_std / (local_var_mean + 1e-10)

        logger.info(
            f"Noise analysis: variance={noise_variance:.2f}, "
            f"consistency={noise_consistency:.4f}"
        )

        return {
            "noise_variance": float(noise_variance),
            "local_variance_mean": float(local_var_mean),
            "noise_consistency": float(noise_consistency),
            # UPDATED: More sensitive threshold (was 0.3, now 0.45)
            "suspicious": bool(noise_consistency < 0.45)
        }

    def analyze_frequency_domain(self) -> Dict[str, Any]:
        """
        Analyze frequency domain via 2D FFT.

        Mathematical basis:
            F(u,v) = Σ I(x,y) · e^(-j2π(ux+vy))
            Ratio = LowFreqEnergy / HighFreqEnergy
            H(X) = -Σ p(x)log p(x)  (spectral entropy)

        Real photos:  Energy decays gradually with frequency
        AI images:    Abnormal high-frequency spikes or flat spectrum

        Returns:
            Dictionary with frequency metrics
        """
        # 2D Fast Fourier Transform
        f_transform = fft.fft2(self.cv_gray)
        f_shift = fft.fftshift(f_transform)  # Zero frequency to center
        magnitude_spectrum = np.abs(f_shift)

        rows, cols = self.cv_gray.shape
        crow, ccol = rows // 2, cols // 2

        # Safe center_size for small images
        center_size = min(30, crow, ccol)

        # Low freq = center region, High freq = everything else
        low_freq = magnitude_spectrum[
            crow - center_size:crow + center_size,
            ccol - center_size:ccol + center_size
        ].sum()
        high_freq = magnitude_spectrum.sum() - low_freq
        freq_ratio = low_freq / (high_freq + 1e-10)

        # Spectral entropy: lower = less natural frequency distribution
        spectrum_flat = magnitude_spectrum.flatten()
        spectrum_normalized = spectrum_flat / (spectrum_flat.sum() + 1e-10)
        spectral_entropy = float(entropy(spectrum_normalized + 1e-10))

        logger.info(
            f"Frequency analysis: ratio={freq_ratio:.4f}, "
            f"entropy={spectral_entropy:.2f}"
        )

        return {
            "frequency_ratio": float(freq_ratio),
            "spectral_entropy": spectral_entropy,
            # UPDATED: More sensitive threshold (was 15.0, now 8.0)
            "suspicious": bool(freq_ratio > 8.0)
        }

    def analyze_jpeg_artifacts(self) -> Dict[str, Any]:
        """
        Analyze JPEG DCT block boundary artifacts.

        Mathematical basis:
            JPEG uses 8x8 DCT blocks with quantization
            Block discontinuity = boundary artifact strength

        Real photos:  Authentic JPEG compression boundary patterns
        AI images:    Often over-smoothed or lack realistic artifacts

        Returns:
            Dictionary with JPEG metrics
        """
        blockiness_scores = []

        for i in range(0, self.cv_gray.shape[0] - 8, 8):
            for j in range(0, self.cv_gray.shape[1] - 8, 8):
                block = self.cv_gray[i:i + 8, j:j + 8].astype(float)
                v_diff = np.abs(block[:, 7] - block[:, 0]).mean()
                h_diff = np.abs(block[7, :] - block[0, :]).mean()
                blockiness_scores.append(v_diff + h_diff)

        # Guard against NaN when image smaller than 8x8
        blockiness = float(np.mean(blockiness_scores)) if blockiness_scores else 0.0

        # Edge density: lower = smoother = more suspicious
        edges = cv2.Canny(self.cv_gray, 100, 200)
        # True edge density: fraction of pixels that are edges (0.0-1.0)
        edge_density = float(np.count_nonzero(edges) / edges.size)

        logger.info(
            f"JPEG analysis: blockiness={blockiness:.2f}, "
            f"edge_density={edge_density:.6f}"
        )

        return {
            "blockiness": blockiness,
            "edge_density": edge_density,
            # UPDATED: More sensitive thresholds (was 2.0/0.01, now 3.5/0.015)
            "suspicious": bool(blockiness < 3.5 or edge_density < 0.015)
        }

    def analyze_color_distribution(self) -> Dict[str, Any]:
        """
        Analyze color distribution via HSV histogram entropy.

        Mathematical basis:
            H(X) = -Σ p(x)log p(x)  applied to hue histogram
            Lower entropy = less color diversity = more suspicious

        Real photos:  Natural color variance and distribution
        AI images:    Sometimes oversaturated or unnaturally uniform

        Returns:
            Dictionary with color metrics
        """
        # HSV separates color (H), saturation (S), brightness (V)
        hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

        h_var = float(hsv[:, :, 0].var())
        s_var = float(hsv[:, :, 1].var())
        v_var = float(hsv[:, :, 2].var())

        # Hue histogram entropy
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_normalized = hist_h / (hist_h.sum() + 1e-10)
        color_entropy = float(entropy(hist_normalized.flatten() + 1e-10))

        mean_saturation = float(hsv[:, :, 1].mean())

        logger.info(
            f"Color analysis: entropy={color_entropy:.2f}, "
            f"sat={mean_saturation:.2f}"
        )

        return {
            "hue_variance": h_var,
            "saturation_variance": s_var,
            "value_variance": v_var,
            "color_entropy": color_entropy,
            "mean_saturation": mean_saturation,
            "suspicious": bool(mean_saturation > 150)
        }

    def calculate_ai_probability(self, signals: Dict[str, Dict]) -> float:
        """
        Combine detection signals into single probability score.

        Weighted ensemble of normalized signals.
        Weights reflect empirical reliability of each signal.

        Args:
            signals: All detection signal dictionaries

        Returns:
            float: AI probability 0.0 (authentic) → 1.0 (AI-generated)
        """
        suspicious_count = sum([
            signals["noise"]["suspicious"],
            signals["frequency"]["suspicious"],
            signals["jpeg"]["suspicious"],
            signals["color"]["suspicious"]
        ])

        weights = {
            "noise_consistency": 0.25,
            "frequency_ratio":   0.25,
            "blockiness":        0.20,
            "color_entropy":     0.15,
            "edge_density":      0.15
        }

        # Normalize each signal to [0, 1] where 1 = most suspicious
        normalized_scores = {
            "noise_consistency": max(0.0, 1.0 - signals["noise"]["noise_consistency"] / 0.5),
            "frequency_ratio":   min(1.0, signals["frequency"]["frequency_ratio"] / 20.0),
            "blockiness":        max(0.0, 1.0 - signals["jpeg"]["blockiness"] / 5.0),
            "color_entropy":     max(0.0, 1.0 - signals["color"]["color_entropy"] / 5.0),
            "edge_density":      max(0.0, 1.0 - signals["jpeg"]["edge_density"] / 0.05)
        }

        probability = sum(
            score * weights[name]
            for name, score in normalized_scores.items()
        )

        # UPDATED: More aggressive boosting when signals agree
        # Boost if 2+ signals agree (was 3+)
        if suspicious_count >= 2:
            probability = min(1.0, probability * 1.3)  # Was 1.2
        # Extra boost if 3+ signals agree
        if suspicious_count >= 3:
            probability = min(1.0, probability * 1.5)  # Additional boost

        logger.info(
            f"AI probability: {probability:.3f} "
            f"({suspicious_count}/4 signals suspicious)"
        )

        return float(probability)

    def detect(self) -> Dict[str, Any]:
        """
        Run complete AI detection pipeline.

        Returns:
            Comprehensive detection report as JSON-serializable dict
        """
        logger.info(f"Starting AI detection for {self.filename}")

        # Run all 4 independent detection signals
        noise_signals = self.analyze_noise_patterns()
        freq_signals  = self.analyze_frequency_domain()
        jpeg_signals  = self.analyze_jpeg_artifacts()
        color_signals = self.analyze_color_distribution()

        all_signals = {
            "noise":     noise_signals,
            "frequency": freq_signals,
            "jpeg":      jpeg_signals,
            "color":     color_signals
        }

        ai_probability = self.calculate_ai_probability(all_signals)

        # Classify based on probability threshold
        if ai_probability > 0.7:
            classification = "likely_ai_generated"
            confidence     = "high"
        elif ai_probability > 0.4:
            classification = "possibly_ai_generated"
            confidence     = "medium"
        else:
            classification = "likely_authentic"
            confidence     = "high" if ai_probability < 0.2 else "medium"

        report = {
            "ai_probability": ai_probability,
            "classification": classification,
            "confidence": confidence,
            "detection_signals": all_signals,
            "summary": {
                "suspicious_signals_count": int(sum(
                    s["suspicious"] for s in all_signals.values()
                )),
                "total_signals": len(all_signals)
            }
        }

        logger.info(
            f"Detection complete: {classification} "
            f"(probability={ai_probability:.3f})"
        )

        return report
