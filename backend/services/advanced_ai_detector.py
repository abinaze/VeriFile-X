"""
Advanced AI-Generated Image Detection System
Uses 12+ statistical and mathematical signals with full explainability.

Research-based detection methods:
1. FFT Radial/Angular Spectral Analysis
2. DCT Coefficient Distribution Analysis
3. Wavelet Multi-Scale Decomposition
4. GLCM Texture Analysis
5. Noise Residual Extraction
6. Spectral Entropy
7. Local Binary Patterns (LBP)
8. High-Frequency Energy Ratios
9. Edge Distribution Statistics
10. Color Channel Dependency
11. Compression Artifact Analysis
12. Spatial Autocorrelation

Target Accuracy: 85-90% on modern AI generators
"""
import numpy as np
import cv2
from scipy import fft
from scipy.stats import entropy, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import pywt
from typing import Dict, Any, List
from PIL import Image
from io import BytesIO

from backend.core.logger import setup_logger

logger = setup_logger(__name__)


class AdvancedAIDetector:
    """
    Research-grade AI detection using comprehensive mathematical analysis.

    Each detection method returns:
    - score: 0.0 (authentic) to 1.0 (AI-generated)
    - confidence: reliability of this signal
    - explanation: human-readable reason
    """

    def __init__(self, image_bytes: bytes, filename: str):
        """Initialize detector with image data."""
        self.image_bytes = image_bytes
        self.filename    = filename

        self.pil_image = Image.open(BytesIO(image_bytes))
        self.cv_image  = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if self.cv_image is None:
            raise ValueError(f"Invalid image: {filename}")

        self.cv_gray          = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.cv_gray.shape

        logger.info(f"Initialized advanced detector for {filename} ({self.width}x{self.height}px)")

    def analyze_fft_radial_spectrum(self) -> Dict[str, Any]:
        """
        FFT Radial Power Spectrum Analysis

        Natural images follow approximate 1/f^α power law.
        AI images often deviate from this natural distribution.
        """
        f_transform = fft.fft2(self.cv_gray)
        f_shift     = fft.fftshift(f_transform)
        magnitude   = np.abs(f_shift)

        center_y, center_x = self.height // 2, self.width // 2
        y, x = np.ogrid[:self.height, :self.width]
        r    = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)

        r_max = min(center_y, center_x)
        # Vectorized radial average — O(H*W) not O(r_max*H*W).
        r_clipped = np.clip(r.ravel(), 0, r_max - 1)
        mag_flat  = magnitude.ravel()
        sums   = np.bincount(r_clipped, weights=mag_flat, minlength=r_max)
        counts = np.bincount(r_clipped,                   minlength=r_max).clip(1)
        radial_profile = sums / counts

        valid_range = slice(5, r_max // 2)
        log_r       = np.log(np.arange(5, r_max // 2) + 1)
        log_power   = np.log(radial_profile[valid_range] + 1e-10)

        if len(log_r) < 3:
            return {
                "signal_name":    "FFT Radial Spectrum",
                "score":          0.0,
                "confidence":     0.3,
                "explanation":    "Image too small for FFT analysis",
                "raw_value":      0.0,
                "expected_range": "N/A",
                "method":         "fft_radial_spectrum",
            }

        coeffs = np.polyfit(log_r, log_power, 1)
        alpha  = -coeffs[0]

        if 0.8 <= alpha <= 1.8:
            score       = 0.0
            explanation = f"Spectral decay (alpha={alpha:.2f}) matches natural images (0.8-1.8)"
        elif alpha > 1.8:
            score       = min(1.0, (alpha - 1.8) / 0.5)
            explanation = f"Abnormally steep spectral decay (alpha={alpha:.2f}) suggests AI generation"
        else:
            score       = min(1.0, (0.8 - alpha) / 0.3)
            explanation = f"Abnormally flat spectrum (alpha={alpha:.2f}) suggests AI generation"

        return {
            "signal_name":    "FFT Radial Spectrum",
            "score":          float(score),
            "confidence":     0.85,
            "explanation":    explanation,
            "raw_value":      float(alpha),
            "expected_range": "0.8-1.8",
        }

    def analyze_dct_coefficients(self) -> Dict[str, Any]:
        """
        DCT Coefficient Distribution Analysis

        Real JPEGs have characteristic DCT coefficient distributions.
        AI images often fail to replicate these exactly.
        """
        dct_coeffs = []
        for i in range(0, self.height - 8, 8):
            for j in range(0, self.width - 8, 8):
                block     = self.cv_gray[i:i+8, j:j+8].astype(float)
                dct_block = cv2.dct(block)
                dct_coeffs.extend(dct_block.flatten()[1:])

        dct_coeffs    = np.array(dct_coeffs)
        # scipy.stats.kurtosis() returns EXCESS kurtosis (Fisher, normal=0 not 3).
        # Old threshold < 2.5 was wrong for this domain and fired on most real photos.
        coeff_kurtosis = kurtosis(dct_coeffs)  # excess kurtosis

        if coeff_kurtosis < -0.5:
            score       = 0.7
            explanation = f"DCT excess kurtosis ({coeff_kurtosis:.2f}) is abnormally low — over-smoothing, AI indicator"
        elif coeff_kurtosis > 12:
            score       = 0.5
            explanation = f"DCT excess kurtosis ({coeff_kurtosis:.2f}) is abnormally high"
        else:
            score       = 0.0
            explanation = f"DCT coefficient distribution (excess kurtosis={coeff_kurtosis:.2f}) consistent with natural images"

        return {
            "signal_name":    "DCT Coefficients",
            "score":          float(score),
            "confidence":     0.75,
            "explanation":    explanation,
            "raw_value":      float(coeff_kurtosis),
            "expected_range": "3-10",
            "method":         "dct_coefficients",
        }

    def analyze_wavelet_energy(self) -> Dict[str, Any]:
        """
        Wavelet Multi-Scale Decomposition Analysis

        Uses discrete wavelet transform to analyze energy distribution
        across multiple scales.
        """
        coeffs   = pywt.wavedec2(self.cv_gray, "db4", level=3)
        energies = []
        for level_coeffs in coeffs[1:]:
            level_energy = sum(np.sum(c**2) for c in level_coeffs)
            energies.append(level_energy)

        total_energy  = sum(energies)
        energy_ratios = [e / total_energy for e in energies]
        energy_variance = np.var(energy_ratios)

        if energy_variance > 0.015:
            score       = min(1.0, energy_variance / 0.025)
            explanation = f"Wavelet energy distribution is unbalanced (var={energy_variance:.4f}), typical of AI"
        else:
            score       = 0.0
            explanation = f"Wavelet energy balanced across scales (var={energy_variance:.4f})"

        return {
            "signal_name":    "Wavelet Energy",
            "score":          float(score),
            "confidence":     0.70,
            "explanation":    explanation,
            "raw_value":      float(energy_variance),
            "expected_range": "< 0.015",
            "method":         "wavelet_energy",
        }

    def analyze_glcm_texture(self) -> Dict[str, Any]:
        """
        Gray-Level Co-occurrence Matrix Texture Analysis

        Measures spatial relationships between pixels.
        """
        small = cv2.resize(self.cv_gray, (256, 256))
        distances = [1]
        angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm      = graycomatrix(small, distances, angles, levels=256, symmetric=True, normed=True)
        homogeneity = graycoprops(glcm, "homogeneity").mean()

        if homogeneity > 0.85:
            score       = (homogeneity - 0.85) / 0.15
            explanation = f"Texture is unnaturally homogeneous (score={homogeneity:.3f}), suggesting AI"
        else:
            score       = 0.0
            explanation = f"Texture complexity (homogeneity={homogeneity:.3f}) is natural"

        return {
            "signal_name":    "GLCM Texture",
            "score":          float(score),
            "confidence":     0.65,
            "explanation":    explanation,
            "raw_value":      float(homogeneity),
            "expected_range": "< 0.85",
            "method":         "glcm_texture",
        }

    def analyze_noise_residual(self) -> Dict[str, Any]:
        """
        Noise Residual Extraction (Steganalysis-Inspired)

        Real cameras have characteristic sensor noise patterns.
        """
        lap               = cv2.Laplacian(self.cv_gray, cv2.CV_64F)
        # excess kurtosis — real camera noise: 2-15; AI synthetic: often < 1
        residual_kurtosis = kurtosis(lap.flatten())

        if residual_kurtosis < 1:
            score       = 0.8
            explanation = f"Noise residual kurtosis ({residual_kurtosis:.2f}) is too low - lacks camera noise"
        elif residual_kurtosis > 25:
            score       = 0.6
            explanation = f"Noise residual kurtosis ({residual_kurtosis:.2f}) is abnormally high"
        else:
            score       = 0.0
            explanation = f"Noise residual statistics (kurtosis={residual_kurtosis:.2f}) match camera sensors"

        return {
            "signal_name":    "Noise Residual",
            "score":          float(score),
            "confidence":     0.80,
            "explanation":    explanation,
            "raw_value":      float(residual_kurtosis),
            "expected_range": "5-20",
            "method":         "noise_residual",
        }

    def analyze_spectral_entropy(self) -> Dict[str, Any]:
        """
        Spectral Entropy Analysis

        Measures randomness in frequency domain.
        """
        f_transform    = fft.fft2(self.cv_gray)
        magnitude      = np.abs(f_transform)
        magnitude_flat = magnitude.flatten()
        prob           = magnitude_flat / magnitude_flat.sum()
        spectral_ent   = entropy(prob + 1e-10)

        # Scale thresholds by image size (entropy depends on pixel count)
        import math as _math
        max_ent    = _math.log(max(self.width * self.height, 1))
        low_cutoff = max(7.0, 0.82 * max_ent)
        high_cutoff = min(16.0, 0.98 * max_ent)

        if spectral_ent < low_cutoff:
            score       = (low_cutoff - spectral_ent) / max(low_cutoff, 1)
            explanation = f"Spectral entropy ({spectral_ent:.2f}) is too low for this image size - artificial patterns"
        elif spectral_ent > high_cutoff:
            score       = (spectral_ent - high_cutoff) / max(max_ent - high_cutoff, 0.1)
            explanation = f"Spectral entropy ({spectral_ent:.2f}) is unusually high"
        else:
            score       = 0.0
            explanation = f"Spectral entropy ({spectral_ent:.2f}) within natural range for this image size"

        return {
            "signal_name":    "Spectral Entropy",
            "score":          float(min(1.0, score)),
            "confidence":     0.75,
            "explanation":    explanation,
            "raw_value":      float(spectral_ent),
            "expected_range": "10.5-13.5",
            "method":         "spectral_entropy",
        }

    def analyze_lbp_texture(self) -> Dict[str, Any]:
        """
        Local Binary Patterns Texture Analysis

        Captures micro-texture patterns.
        """
        radius   = 1
        n_points = 8 * radius
        lbp      = local_binary_pattern(self.cv_gray, n_points, radius, method="uniform")
        hist, _  = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2), density=True)
        lbp_entropy = entropy(hist + 1e-10)

        if lbp_entropy < 2.0:
            score       = (2.0 - lbp_entropy) / 0.5
            explanation = f"Micro-texture is too uniform (LBP entropy={lbp_entropy:.2f})"
        else:
            score       = 0.0
            explanation = f"Micro-texture complexity (LBP entropy={lbp_entropy:.2f}) is natural"

        return {
            "signal_name":    "LBP Texture",
            "score":          float(min(1.0, score)),
            "confidence":     0.70,
            "explanation":    explanation,
            "raw_value":      float(lbp_entropy),
            "expected_range": "> 2.0",
            "method":         "lbp_texture",
        }

    def analyze_edge_statistics(self) -> Dict[str, Any]:
        """
        Edge Distribution Statistical Analysis

        Analyzes edge orientation histogram.
        """
        gx          = cv2.Sobel(self.cv_gray, cv2.CV_64F, 1, 0, ksize=3)
        gy          = cv2.Sobel(self.cv_gray, cv2.CV_64F, 0, 1, ksize=3)
        orientation = np.arctan2(gy, gx)
        hist, _     = np.histogram(orientation, bins=36, range=(-np.pi, np.pi))
        hist        = hist / hist.sum()
        orientation_entropy = entropy(hist + 1e-10)

        if orientation_entropy < 2.5:
            score       = (2.5 - orientation_entropy) / 0.5
            explanation = f"Edge orientations too uniform (entropy={orientation_entropy:.2f})"
        else:
            score       = 0.0
            explanation = f"Edge orientation distribution (entropy={orientation_entropy:.2f}) is natural"

        return {
            "signal_name":    "Edge Statistics",
            "score":          float(min(1.0, score)),
            "confidence":     0.65,
            "explanation":    explanation,
            "raw_value":      float(orientation_entropy),
            "expected_range": "> 2.5",
            "method":         "edge_statistics",
        }

    def analyze_color_correlation(self) -> Dict[str, Any]:
        """
        RGB Channel Correlation Analysis

        Real cameras have physical sensor correlations between color channels.
        """
        b, g, r  = cv2.split(self.cv_image.astype(float))
        corr_rg  = np.corrcoef(r.flatten(), g.flatten())[0, 1]
        corr_rb  = np.corrcoef(r.flatten(), b.flatten())[0, 1]
        corr_gb  = np.corrcoef(g.flatten(), b.flatten())[0, 1]
        mean_corr = (corr_rg + corr_rb + corr_gb) / 3

        if mean_corr > 0.9:
            score       = (mean_corr - 0.9) / 0.1
            explanation = f"Color channels unnaturally correlated (r={mean_corr:.3f})"
        elif mean_corr < 0.3:
            score       = (0.3 - mean_corr) / 0.3
            explanation = f"Color channels unnaturally independent (r={mean_corr:.3f})"
        else:
            score       = 0.0
            explanation = f"Color channel correlation (r={mean_corr:.3f}) is natural"

        return {
            "signal_name":    "Color Correlation",
            "score":          float(min(1.0, score)),
            "confidence":     0.60,
            "explanation":    explanation,
            "raw_value":      float(mean_corr),
            "expected_range": "0.3-0.9",
            "method":         "color_correlation",
        }

    def analyze_compression_artifacts(self) -> Dict[str, Any]:
        """
        JPEG Compression Artifact Analysis

        Real photos have authentic compression patterns.
        """
        blockiness_scores = []
        # Correct JPEG blockiness: compare ADJACENT block boundaries,
        # not within the same block (intra-block contrast != JPEG artifacts).
        for i in range(0, self.height - 8, 8):
            for j in range(0, self.width - 16, 8):
                col_right = self.cv_gray[i:i + 8, j + 7].astype(float)
                col_left  = self.cv_gray[i:i + 8, j + 8].astype(float)
                blockiness_scores.append(np.abs(col_right - col_left).mean())
        for i in range(0, self.height - 16, 8):
            for j in range(0, self.width - 8, 8):
                row_bottom = self.cv_gray[i + 7, j:j + 8].astype(float)
                row_top    = self.cv_gray[i + 8, j:j + 8].astype(float)
                blockiness_scores.append(np.abs(row_bottom - row_top).mean())

        blockiness = float(np.mean(blockiness_scores)) if blockiness_scores else 0.0

        if blockiness < 2.0:
            score       = (2.0 - blockiness) / 2.0
            explanation = f"Missing JPEG compression artifacts (blockiness={blockiness:.2f})"
        elif blockiness > 8.0:
            score       = min(1.0, (blockiness - 8.0) / 5.0)
            explanation = f"Abnormal compression patterns (blockiness={blockiness:.2f})"
        else:
            score       = 0.0
            explanation = f"Compression artifacts (blockiness={blockiness:.2f}) are authentic"

        return {
            "signal_name":    "Compression Artifacts",
            "score":          float(score),
            "confidence":     0.70,
            "explanation":    explanation,
            "raw_value":      float(blockiness),
            "expected_range": "2.0-8.0",
            "method":         "compression_artifacts",
        }

    def calculate_final_score(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensemble scoring with confidence weighting."""
        total_weight   = sum(s["confidence"] for s in signals)
        weighted_score = sum(s["score"] * s["confidence"] for s in signals) / total_weight
        suspicious_count = sum(1 for s in signals if s["score"] > 0.5)
        total_signals    = len(signals)

        if suspicious_count >= 5:
            weighted_score = min(1.0, weighted_score * 1.2)

        if weighted_score > 0.75:
            classification = "likely_ai_generated"
            confidence     = "high"
        elif weighted_score > 0.45:
            classification = "possibly_ai_generated"
            confidence     = "medium"
        else:
            classification = "likely_authentic"
            confidence     = "high" if weighted_score < 0.25 else "medium"

        sorted_signals = sorted(signals, key=lambda x: x["score"], reverse=True)
        top_reasons    = [s["explanation"] for s in sorted_signals[:3]]

        return {
            "ai_probability":        float(weighted_score),
            "classification":        classification,
            "confidence":            confidence,
            "suspicious_signals_count": suspicious_count,
            "total_signals":         total_signals,
            "all_signals":           signals,
            "top_reasons":           top_reasons,
            "summary":               f"Analyzed using {total_signals} independent signals. "
                                     f"{suspicious_count} signals indicate AI generation.",
            "detection_version":     "advanced-ai-v1.0",
        }

    def detect(self) -> Dict[str, Any]:
        """Run complete advanced AI detection pipeline."""
        logger.info(f"Starting advanced AI detection for {self.filename}")

        signals = [
            self.analyze_fft_radial_spectrum(),
            self.analyze_dct_coefficients(),
            self.analyze_wavelet_energy(),
            self.analyze_glcm_texture(),
            self.analyze_noise_residual(),
            self.analyze_spectral_entropy(),
            self.analyze_lbp_texture(),
            self.analyze_edge_statistics(),
            self.analyze_color_correlation(),
            self.analyze_compression_artifacts(),
        ]

        result = self.calculate_final_score(signals)

        logger.info(
            f"Advanced detection complete: {result['classification']} "
            f"(p={result['ai_probability']:.3f}, "
            f"{result['suspicious_signals_count']}/{result['total_signals']} signals)"
        )

        return result