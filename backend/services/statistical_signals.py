"""
Composable statistical/frequency-domain signal providers (F-16).

Replaces the previous 5-level inheritance chain
(AdvancedAIDetector -> UltraAdvancedDetector -> CovarianceDetector ->
StatisticalDetector -> AdvancedEnsembleDetector) with composition: each
of the 4 original signal groups below is now an independent component
operating on a shared, explicit ImageContext, instead of a class that
inherits -- and therefore silently depends on internal attributes of
-- the one "below" it in the chain. This is a pure architectural
change: every signal-computation method's logic is unchanged from the
original (verified via a byte-for-byte behavioral comparison across
real test images before this replaced the inheritance chain), only
`self.X` -> `self.ctx.X` for the shared decoded-image state.

AdvancedAIDetector / UltraAdvancedDetector / CovarianceDetector /
StatisticalDetector (backend/services/advanced_ai_detector.py etc.)
are kept as backward-compatible facades over these components --
several existing tests construct and call those classes, and some
(test_signal_quality.py) call methods several "levels" up the old
inheritance chain directly on a StatisticalDetector instance (e.g.
.analyze_fft_radial_spectrum(), originally only defined on the base
class four levels down) -- so those facades compose ALL of the
components below their old position in the chain and re-expose every
method, not just their own new ones.
"""
import threading
import numpy as np
import cv2
from scipy import fft
from scipy.stats import entropy, kurtosis
from scipy.linalg import eig
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import pywt
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from PIL import Image
from io import BytesIO

from backend.core.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ImageContext:
    """Shared, read-only decoded-image state every signal provider
    needs. Replaces what used to be instance attributes set up once by
    AdvancedAIDetector.__init__ (self.cv_gray, self.height, ...) and
    silently relied on by every method throughout the whole 5-level
    inheritance chain.
    """
    image_bytes: bytes
    filename:    str
    pil_image:   Image.Image
    cv_image:    np.ndarray
    cv_gray:     np.ndarray
    height:      int
    width:       int

    @classmethod
    def from_bytes(cls, image_bytes: bytes, filename: str) -> "ImageContext":
        pil_image = Image.open(BytesIO(image_bytes))
        cv_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if cv_image is None:
            raise ValueError(f"Invalid image: {filename}")
        cv_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        height, width = cv_gray.shape
        return cls(
            image_bytes=image_bytes, filename=filename, pil_image=pil_image,
            cv_image=cv_image, cv_gray=cv_gray, height=height, width=width,
        )


def score_and_classify(
    signals: List[Dict[str, Any]],
    boost_schedule: List[Tuple[int, float]],
    detection_version: str,
) -> Dict[str, Any]:
    """Shared confidence-weighted scoring + classification, used by all
    four facade classes' detect() methods (F-16) -- previously each of
    the 4 inheritance levels carried an almost-identical copy of this
    exact computation (confidence-weighted average, a suspicious-count
    boost multiplier, classification thresholds, top_reasons, summary),
    differing only in the boost thresholds and the detection_version
    string. One implementation instead of four.

    Args:
        signals: full signal list (e.g. 10, 13, 16, or 19 entries).
        boost_schedule: list of (min_suspicious_count, multiplier) pairs,
            checked in the given order; the FIRST matching threshold
            wins (callers must list higher thresholds first -- matches
            the original elif-chain semantics exactly). E.g.
            [(10, 1.35), (8, 1.25), (6, 1.15)].
        detection_version: the report's "detection_version" string.
    """
    total_weight   = sum(s["confidence"] for s in signals)
    weighted_score = sum(s["score"] * s["confidence"] for s in signals) / total_weight
    suspicious_count = sum(1 for s in signals if s["score"] > 0.5)
    total_signals    = len(signals)

    for threshold, multiplier in boost_schedule:
        if suspicious_count >= threshold:
            weighted_score = min(1.0, weighted_score * multiplier)
            break

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
        "ai_probability":           float(weighted_score),
        "classification":           classification,
        "confidence":               confidence,
        "suspicious_signals_count": suspicious_count,
        "total_signals":            total_signals,
        "all_signals":              signals,
        "top_reasons":              top_reasons,
        "summary": (
            f"Analyzed using {total_signals} independent signals. "
            f"{suspicious_count} signals indicate AI generation."
        ),
        "detection_version": detection_version,
    }


class BasicSignals:
    """The original 10 signals from AdvancedAIDetector: FFT radial
    spectrum, DCT coefficients, wavelet energy, GLCM texture, noise
    residual, spectral entropy, LBP texture, edge statistics, color
    correlation, compression artifacts."""

    def __init__(self, ctx: ImageContext):
        self.ctx = ctx

    def analyze_fft_radial_spectrum(self) -> Dict[str, Any]:
        """
        FFT Radial Power Spectrum Analysis

        Natural images follow approximate 1/f^α power law.
        AI images often deviate from this natural distribution.
        """
        f_transform = fft.fft2(self.ctx.cv_gray)
        f_shift     = fft.fftshift(f_transform)
        magnitude   = np.abs(f_shift)

        center_y, center_x = self.ctx.height // 2, self.ctx.width // 2
        y, x = np.ogrid[:self.ctx.height, :self.ctx.width]
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
        for i in range(0, self.ctx.height - 8, 8):
            for j in range(0, self.ctx.width - 8, 8):
                block     = self.ctx.cv_gray[i:i+8, j:j+8].astype(float)
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
        coeffs   = pywt.wavedec2(self.ctx.cv_gray, "db4", level=3)
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
        small = cv2.resize(self.ctx.cv_gray, (256, 256))
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
        lap               = cv2.Laplacian(self.ctx.cv_gray, cv2.CV_64F)
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
        f_transform    = fft.fft2(self.ctx.cv_gray)
        magnitude      = np.abs(f_transform)
        magnitude_flat = magnitude.flatten()
        prob           = magnitude_flat / magnitude_flat.sum()
        spectral_ent   = entropy(prob + 1e-10)

        # Scale thresholds by image size (entropy depends on pixel count)
        import math as _math
        max_ent    = _math.log(max(self.ctx.width * self.ctx.height, 1))
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
        lbp      = local_binary_pattern(self.ctx.cv_gray, n_points, radius, method="uniform")
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
        gx          = cv2.Sobel(self.ctx.cv_gray, cv2.CV_64F, 1, 0, ksize=3)
        gy          = cv2.Sobel(self.ctx.cv_gray, cv2.CV_64F, 0, 1, ksize=3)
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
        b, g, r  = cv2.split(self.ctx.cv_image.astype(float))
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
        for i in range(0, self.ctx.height - 8, 8):
            for j in range(0, self.ctx.width - 16, 8):
                col_right = self.ctx.cv_gray[i:i + 8, j + 7].astype(float)
                col_left  = self.ctx.cv_gray[i:i + 8, j + 8].astype(float)
                blockiness_scores.append(np.abs(col_right - col_left).mean())
        for i in range(0, self.ctx.height - 16, 8):
            for j in range(0, self.ctx.width - 8, 8):
                row_bottom = self.ctx.cv_gray[i + 7, j:j + 8].astype(float)
                row_top    = self.ctx.cv_gray[i + 8, j:j + 8].astype(float)
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


    def compute_all(self) -> List[Dict[str, Any]]:
        return [
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


class UltraSignals:
    """The 3 signals added by UltraAdvancedDetector: cross-channel
    noise covariance, patch-level spectral variance, natural image
    prior deviation."""

    def __init__(self, ctx: ImageContext):
        self.ctx = ctx

    def analyze_rgb_noise_covariance(self) -> Dict[str, Any]:
        """
        Cross-Channel Noise Covariance Analysis

        Real cameras: RGB channels share sensor physics — correlated noise
        AI images: Often independently synthesized — lower correlation
        """
        b, g, r = cv2.split(self.ctx.cv_image.astype(float))

        r_noise = cv2.Laplacian(r, cv2.CV_64F).flatten()
        g_noise = cv2.Laplacian(g, cv2.CV_64F).flatten()
        b_noise = cv2.Laplacian(b, cv2.CV_64F).flatten()

        noise_matrix = np.vstack([r_noise, g_noise, b_noise])
        cov_matrix   = np.corrcoef(noise_matrix)

        rg_corr   = cov_matrix[0, 1]
        rb_corr   = cov_matrix[0, 2]
        gb_corr   = cov_matrix[1, 2]
        mean_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3

        if mean_corr < 0.5:
            score       = (0.5 - mean_corr) / 0.3
            explanation = f"RGB noise correlation ({mean_corr:.3f}) is abnormally low - channels synthesized independently"
        elif mean_corr > 0.85:
            score       = (mean_corr - 0.85) / 0.15
            explanation = f"RGB noise correlation ({mean_corr:.3f}) is unnaturally high"
        else:
            score       = 0.0
            explanation = f"RGB noise correlation ({mean_corr:.3f}) matches camera sensor physics"

        return {
            "signal_name":    "RGB Noise Covariance",
            "score":          float(min(1.0, score)),
            "confidence":     0.88,
            "explanation":    explanation,
            "raw_value":      float(mean_corr),
            "expected_range": "0.5-0.85",
            "method":         "cross_channel_noise_covariance",
        }


    def analyze_patch_spectral_variance(self) -> Dict[str, Any]:
        """
        Patch-Level FFT Variance Analysis

        Research basis: Far more robust than single-image FFT.
        """
        patch_size = 128
        alphas     = []

        for i in range(0, self.ctx.height - patch_size, patch_size):
            for j in range(0, self.ctx.width - patch_size, patch_size):
                patch = self.ctx.cv_gray[i:i+patch_size, j:j+patch_size]

                f_transform    = fft.fft2(patch)
                f_shift        = fft.fftshift(f_transform)
                magnitude      = np.abs(f_shift)
                center_y, center_x = patch_size // 2, patch_size // 2
                y, x = np.ogrid[:patch_size, :patch_size]
                r    = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)

                r_max = patch_size // 4
                # Vectorized radial average — O(patch²) not O(r_max*patch²).
                r_clip = np.clip(r.ravel(), 0, r_max - 1)
                m_flat = magnitude.ravel()
                sums_p   = np.bincount(r_clip, weights=m_flat, minlength=r_max)
                counts_p = np.bincount(r_clip,                  minlength=r_max).clip(1)
                radial_profile = sums_p / counts_p

                valid_range = slice(5, r_max - 5)
                log_r       = np.log(np.arange(5, r_max - 5) + 1)
                log_power   = np.log(radial_profile[valid_range] + 1e-10)

                if len(log_r) >= 3 and len(log_power) >= 3:
                    coeffs = np.polyfit(log_r, log_power, 1)
                    alpha  = -coeffs[0]
                    alphas.append(alpha)

        if len(alphas) < 4:
            return {
                "signal_name":    "Patch Spectral Variance",
                "score":          0.0,
                "confidence":     0.3,
                "explanation":    "Image too small for patch analysis",
                "raw_value":      0.0,
                "expected_range": "N/A",
                "method":         "patch_level_fft_variance",
            }

        alpha_variance = np.var(alphas)

        if alpha_variance < 0.12:
            score       = (0.12 - alpha_variance) / 0.12
            explanation = f"Spectral uniformity across patches ({alpha_variance:.4f}) suggests synthetic generation"
        else:
            score       = 0.0
            explanation = f"Natural spectral variation across patches ({alpha_variance:.4f})"

        return {
            "signal_name":    "Patch Spectral Variance",
            "score":          float(min(1.0, score)),
            "confidence":     0.85,
            "explanation":    explanation,
            "raw_value":      float(alpha_variance),
            "expected_range": "> 0.12",
            "method":         "patch_level_fft_variance",
        }


    def analyze_natural_prior_deviation(self) -> Dict[str, Any]:
        """
        Natural Image Prior Deviation Score

        Measures log-likelihood deviation from 1/f^2 natural prior.
        """
        f_transform = fft.fft2(self.ctx.cv_gray)
        f_shift     = fft.fftshift(f_transform)
        magnitude   = np.abs(f_shift)

        center_y, center_x = self.ctx.height // 2, self.ctx.width // 2
        y, x = np.ogrid[:self.ctx.height, :self.ctx.width]
        r    = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)

        r_max = min(center_y, center_x) // 2
        # Vectorized radial average — O(H*W) not O(r_max*H*W).
        r_clip = np.clip(r.ravel(), 0, r_max - 1)
        m_flat = magnitude.ravel()
        sums_g   = np.bincount(r_clip, weights=m_flat, minlength=r_max)
        counts_g = np.bincount(r_clip,                  minlength=r_max).clip(1)
        radial_profile = sums_g / counts_g

        valid_range = slice(10, r_max - 10)
        log_r       = np.log(np.arange(10, r_max - 10) + 1)
        log_power   = np.log(radial_profile[valid_range] + 1e-10)

        if len(log_r) < 3 or len(log_power) < 3:
            return {
                "signal_name":    "Natural Prior Deviation",
                "score":          0.0,
                "confidence":     0.3,
                "explanation":    "Image too small for spectral analysis",
                "raw_value":      0.0,
                "expected_range": "N/A",
                "method":         "natural_prior_deviation",
            }

        coeffs    = np.polyfit(log_r, log_power, 1)
        alpha     = -coeffs[0]
        deviation = abs(alpha - 2.0)

        if deviation > 0.4:
            score       = min(1.0, deviation / 0.8)
            explanation = f"Spectral slope (alpha={alpha:.3f}) deviates from natural prior (alpha~2.0)"
        else:
            score       = 0.0
            explanation = f"Spectral slope (alpha={alpha:.3f}) follows natural image statistics"

        return {
            "signal_name":    "Natural Prior Deviation",
            "score":          float(score),
            "confidence":     0.80,
            "explanation":    explanation,
            "raw_value":      float(deviation),
            "expected_range": "< 0.4",
            "method":         "natural_image_prior",
        }


    def compute_all(self) -> List[Dict[str, Any]]:
        return [
            self.analyze_rgb_noise_covariance(),
            self.analyze_patch_spectral_variance(),
            self.analyze_natural_prior_deviation(),
        ]


class CovarianceSignals:
    """The 3 signals added by CovarianceDetector: eigenvalue spread,
    local covariance consistency, patch anisotropy variance."""

    def __init__(self, ctx: ImageContext):
        self.ctx = ctx

    def analyze_eigenvalue_spread(self) -> Dict[str, Any]:
        """
        Covariance Matrix Eigenvalue Analysis
        
        Research basis: Real cameras have dominant principal component in noise
        
        Method:
        1. Extract RGB noise residuals using Laplacian
        2. Compute 3×3 covariance matrix
        3. Calculate eigenvalues λ₁, λ₂, λ₃
        4. Measure spread ratio: λ₁ / λ₃
        
        Real cameras: High spread (λ₁ >> λ₃) - dominant shared noise
        AI images: Lower spread - more uniform eigenvalues
        
        Returns:
            Detection signal with score, confidence, explanation
        """
        # Split RGB channels
        b, g, r = cv2.split(self.ctx.cv_image.astype(float))
        
        # Extract high-frequency noise using Laplacian
        r_noise = cv2.Laplacian(r, cv2.CV_64F).flatten()
        g_noise = cv2.Laplacian(g, cv2.CV_64F).flatten()
        b_noise = cv2.Laplacian(b, cv2.CV_64F).flatten()
        
        # Sample for computational efficiency (use 10k pixels max)
        if len(r_noise) > 10000:
            # Use content-derived seed for determinism — same image always
            # produces the same sample even under concurrent requests.
            import hashlib as _hl
            _seed = int(_hl.sha256(self.ctx.image_bytes[:64]).hexdigest()[:8], 16) % (2**31)
            _rng = np.random.default_rng(_seed)
            indices = _rng.choice(len(r_noise), 10000, replace=False)
            r_noise = r_noise[indices]
            g_noise = g_noise[indices]
            b_noise = b_noise[indices]
        
        # Stack into matrix
        noise_matrix = np.vstack([r_noise, g_noise, b_noise])
        
        # Compute covariance matrix (3×3)
        cov_matrix = np.cov(noise_matrix)
        
        # Calculate eigenvalues
        eigenvalues, _ = eig(cov_matrix)
        eigenvalues = np.real(eigenvalues)  # Take real part
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        # Compute spread ratio
        if eigenvalues[2] < 1e-10:
            spread_ratio = 100.0  # Very high spread
        else:
            spread_ratio = eigenvalues[0] / eigenvalues[2]
        
        # Real cameras: spread typically 5-20
        # AI images: often 2-8 (more uniform)
        if spread_ratio < 4.0:
            score = (4.0 - spread_ratio) / 4.0
            explanation = f"Eigenvalue spread ({spread_ratio:.2f}) is low - noise lacks dominant component"
        elif spread_ratio > 25.0:
            score = min(1.0, (spread_ratio - 25.0) / 20.0)
            explanation = f"Eigenvalue spread ({spread_ratio:.2f}) is abnormally high"
        else:
            score = 0.0
            explanation = f"Eigenvalue spread ({spread_ratio:.2f}) matches camera sensor physics"
        
        logger.info(f"Eigenvalue spread: λ₁={eigenvalues[0]:.2f}, λ₃={eigenvalues[2]:.2f}, ratio={spread_ratio:.2f}")
        
        return {
            "signal_name": "Eigenvalue Spread",
            "score": float(min(1.0, score)),
            "confidence": 0.90,
            "explanation": explanation,
            "raw_value": float(spread_ratio),
            "expected_range": "4.0-25.0",
            "method": "covariance_eigenvalue_analysis"
        }
    

    def analyze_local_covariance_consistency(self) -> Dict[str, Any]:
        """
        Local Covariance Consistency Analysis
        
        Research basis: Real cameras have consistent physical covariance patterns
        
        Method:
        1. Divide image into patches (64×64)
        2. Compute RGB noise covariance per patch
        3. Measure variance of covariances across patches
        
        Real cameras: Consistent covariance structure
        AI images: Irregular spatial variation
        
        Returns:
            Detection signal with score, confidence, explanation
        """
        patch_size = 64
        covariances = []
        
        # Divide into patches
        for i in range(0, self.ctx.height - patch_size, patch_size):
            for j in range(0, self.ctx.width - patch_size, patch_size):
                patch = self.ctx.cv_image[i:i+patch_size, j:j+patch_size]
                
                # Extract RGB
                b, g, r = cv2.split(patch.astype(float))
                
                # Noise residuals
                r_noise = cv2.Laplacian(r, cv2.CV_64F).flatten()
                g_noise = cv2.Laplacian(g, cv2.CV_64F).flatten()
                b_noise = cv2.Laplacian(b, cv2.CV_64F).flatten()
                
                # Covariance matrix
                noise_matrix = np.vstack([r_noise, g_noise, b_noise])
                cov_matrix = np.cov(noise_matrix)
                
                # Store mean off-diagonal correlation
                mean_corr = (abs(cov_matrix[0,1]) + abs(cov_matrix[0,2]) + abs(cov_matrix[1,2])) / 3
                covariances.append(mean_corr)
        
        if len(covariances) < 4:
            return {
                "signal_name": "Local Covariance Consistency",
                "score": 0.0,
                "confidence": 0.3,
                "explanation": "Image too small for patch covariance analysis",
                "raw_value": 0.0,
                "expected_range": "N/A",
                "method": "local_covariance_consistency"
            }
        
        # Compute coefficient of variation (CV = std/mean)
        covariances = np.array(covariances)
        mean_cov = covariances.mean()
        std_cov = covariances.std()
        
        if mean_cov < 1e-6:
            cv = 0.0
        else:
            cv = std_cov / mean_cov
        
        # Real cameras: CV typically 0.1-0.4 (consistent)
        # AI images: often > 0.5 (irregular)
        if cv > 0.45:
            score = min(1.0, (cv - 0.45) / 0.3)
            explanation = f"Covariance consistency (CV={cv:.3f}) is irregular across patches"
        else:
            score = 0.0
            explanation = f"Covariance consistency (CV={cv:.3f}) is natural"
        
        logger.info(f"Local covariance: mean={mean_cov:.4f}, std={std_cov:.4f}, CV={cv:.3f}")
        
        return {
            "signal_name": "Local Covariance Consistency",
            "score": float(score),
            "confidence": 0.85,
            "explanation": explanation,
            "raw_value": float(cv),
            "expected_range": "< 0.45",
            "method": "local_covariance_consistency"
        }
    

    def analyze_patch_anisotropy_variance(self) -> Dict[str, Any]:
        """
        Patch-Level Anisotropy Variance Analysis
        
        Research basis: Natural images have varying directional energy
        
        Method:
        1. Divide into patches
        2. Compute directional energy (horizontal vs vertical) per patch
        3. Measure variance of anisotropy across patches
        
        Anisotropy = |E_horizontal - E_vertical| / (E_horizontal + E_vertical)
        
        Real images: High variance (different textures/edges)
        AI images: Lower variance (more isotropic generation)
        
        Returns:
            Detection signal with score, confidence, explanation
        """
        patch_size = 64
        anisotropies = []
        
        # Divide into patches
        for i in range(0, self.ctx.height - patch_size, patch_size):
            for j in range(0, self.ctx.width - patch_size, patch_size):
                patch = self.ctx.cv_gray[i:i+patch_size, j:j+patch_size].astype(float)
                
                # Compute gradients
                grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
                
                # Energy in each direction
                energy_h = np.sum(grad_x ** 2)
                energy_v = np.sum(grad_y ** 2)
                
                # Anisotropy measure
                total_energy = energy_h + energy_v
                if total_energy > 1e-6:
                    anisotropy = abs(energy_h - energy_v) / total_energy
                else:
                    anisotropy = 0.0
                
                anisotropies.append(anisotropy)
        
        if len(anisotropies) < 4:
            return {
                "signal_name": "Patch Anisotropy Variance",
                "score": 0.0,
                "confidence": 0.3,
                "explanation": "Image too small for anisotropy analysis",
                "raw_value": 0.0,
                "expected_range": "N/A",
                "method": "patch_anisotropy_variance"
            }
        
        # Compute variance of anisotropy
        anisotropy_var = np.var(anisotropies)
        
        # Real images: variance typically 0.02-0.08
        # AI images: often < 0.015 (more uniform)
        if anisotropy_var < 0.015:
            score = (0.015 - anisotropy_var) / 0.015
            explanation = f"Anisotropy variance ({anisotropy_var:.4f}) is too low - uniform directionality"
        else:
            score = 0.0
            explanation = f"Anisotropy variance ({anisotropy_var:.4f}) shows natural variation"
        
        logger.info(f"Anisotropy: mean={np.mean(anisotropies):.4f}, var={anisotropy_var:.4f}")
        
        return {
            "signal_name": "Patch Anisotropy Variance",
            "score": float(min(1.0, score)),
            "confidence": 0.82,
            "explanation": explanation,
            "raw_value": float(anisotropy_var),
            "expected_range": "> 0.015",
            "method": "patch_anisotropy_variance"
        }
    

    def compute_all(self) -> List[Dict[str, Any]]:
        return [
            self.analyze_eigenvalue_spread(),
            self.analyze_local_covariance_consistency(),
            self.analyze_patch_anisotropy_variance(),
        ]


class AdvancedSignals:
    """The 3 signals added by StatisticalDetector: Mahalanobis
    distance, KL divergence, perturbation stability. Shares a
    class-level natural-frequency-model cache across all instances
    (unchanged semantics from the original StatisticalDetector class
    attribute) -- capped at 32 entries, keyed by image size."""

    _natural_model_cache: dict = {}
    _natural_model_lock = threading.Lock()

    def __init__(self, ctx: ImageContext):
        self.ctx = ctx

    def _get_radial_spectrum(self) -> np.ndarray:
        """
        Extract radial power spectrum from image.
        
        Returns:
            1D array of radial spectrum values
        """
        # Compute FFT
        f_transform = fft.fft2(self.ctx.cv_gray)
        f_shift = fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Compute radial profile
        center_y, center_x = self.ctx.height // 2, self.ctx.width // 2
        y, x = np.ogrid[:self.ctx.height, :self.ctx.width]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        # Take log of magnitude
        log_magnitude = np.log(magnitude + 1)
        
        # Radial average
        r_max = min(center_y, center_x) // 2
        # Vectorized bincount — O(H*W) instead of O(r_max * H * W).
        r_clipped = np.clip(r.ravel(), 0, r_max - 1)
        lm_flat   = log_magnitude.ravel()
        sums   = np.bincount(r_clipped, weights=lm_flat,  minlength=r_max)
        counts = np.bincount(r_clipped,                    minlength=r_max).clip(1)
        radial_profile = sums / counts

        return radial_profile
    

    def _build_natural_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build simple natural image frequency model.
        
        In production, this would be precomputed from a large dataset.
        Here we use theoretical expectations based on 1/f² law.
        
        Returns:
            (mean_spectrum, covariance_matrix)
        """
        # For a natural image following 1/f^α with α≈2
        # Log spectrum is approximately linear with slope -α
        
        # Use current image size as reference
        r_max = min(self.ctx.height, self.ctx.width) // 4
        
        # Expected mean: log(A) - α*log(f)
        # where A is amplitude, α is slope (≈2 for natural)
        frequencies = np.arange(1, r_max + 1)
        
        # Natural mean spectrum (theoretical)
        natural_mean = 8.0 - 2.0 * np.log(frequencies + 1)
        
        # Natural covariance (simplified - diagonal with decreasing variance)
        natural_std = 1.5 / (1 + np.log(frequencies + 1))
        natural_cov = np.diag(natural_std ** 2)
        
        return natural_mean, natural_cov
    

    def analyze_mahalanobis_distance(self) -> Dict[str, Any]:
        """
        Mahalanobis Distance in Frequency Space
        
        Research basis: Measures how far image spectrum deviates from natural
        
        Method:
        1. Extract radial power spectrum
        2. Build natural image model (μ, Σ)
        3. Compute Mahalanobis distance: D² = (x-μ)ᵀΣ⁻¹(x-μ)
        
        Natural images: D² typically < 50
        AI images: D² often > 100
        
        Returns:
            Detection signal with score, confidence, explanation
        """
        try:
            # Get image spectrum
            spectrum = self._get_radial_spectrum()
            
            # Build natural model
            r_key = min(self.ctx.height, self.ctx.width) // 4
            with AdvancedSignals._natural_model_lock:
                if r_key not in AdvancedSignals._natural_model_cache:
                    # Evict oldest entry when cache exceeds 32 entries
                    if len(AdvancedSignals._natural_model_cache) >= 32:
                        _evict_key = next(iter(AdvancedSignals._natural_model_cache))
                        del AdvancedSignals._natural_model_cache[_evict_key]
                    AdvancedSignals._natural_model_cache[r_key] = self._build_natural_model()
                natural_mean, natural_cov = AdvancedSignals._natural_model_cache[r_key]
            
            # Ensure same length
            min_len = min(len(spectrum), len(natural_mean))
            spectrum = spectrum[:min_len]
            natural_mean = natural_mean[:min_len]
            natural_cov = natural_cov[:min_len, :min_len]
            
            # Add regularization to covariance
            natural_cov += np.eye(min_len) * 1e-6
            
            # Compute Mahalanobis distance
            try:
                cov_inv = np.linalg.inv(natural_cov)
                diff = spectrum - natural_mean
                mahal_dist = np.sqrt(diff @ cov_inv @ diff)
            except np.linalg.LinAlgError:
                # If inversion fails, use simplified distance
                mahal_dist = np.sqrt(np.sum((spectrum - natural_mean)**2))
            
            # Natural: D² typically 20-80
            # AI: often 100+
            if mahal_dist > 80:
                score = min(1.0, (mahal_dist - 80) / 100)
                explanation = f"Mahalanobis distance ({mahal_dist:.1f}) indicates deviation from natural spectrum"
            else:
                score = 0.0
                explanation = f"Mahalanobis distance ({mahal_dist:.1f}) within natural range"
            
            logger.info(f"Mahalanobis distance: {mahal_dist:.2f}")
            
            return {
                "signal_name": "Mahalanobis Distance",
                "score": float(score),
                "confidence": 0.92,
                "explanation": explanation,
                "raw_value": float(mahal_dist),
                "expected_range": "< 80",
                "method": "mahalanobis_frequency_distance"
            }
            
        except Exception as e:
            logger.warning(f"Mahalanobis analysis failed: {e}")
            return {
                "signal_name": "Mahalanobis Distance",
                "score": 0.0,
                "confidence": 0.3,
                "explanation": "Analysis failed - insufficient data",
                "raw_value": 0.0,
                "expected_range": "N/A",
                "method": "mahalanobis_frequency_distance"
            }
    

    def analyze_kl_divergence(self) -> Dict[str, Any]:
        """
        KL Divergence from Natural Image Prior
        
        Research basis: Measures distribution deviation
        
        Method:
        1. Extract frequency magnitude distribution
        2. Compute natural prior distribution (1/f²)
        3. Calculate D_KL(P_image || P_natural) = Σ P(x) log(P(x)/Q(x))
        
        Natural images: D_KL typically < 0.5
        AI images: often > 1.0
        
        Returns:
            Detection signal with score, confidence, explanation
        """
        try:
            # Get magnitude spectrum — fftshift moves DC to center
            # (without fftshift the DC component is at the corner, corrupting
            # the histogram distribution that feeds the KL divergence).
            f_transform = fft.fft2(self.ctx.cv_gray)
            f_shift = fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Flatten and normalize to probability distribution
            flat_mag = magnitude.flatten()
            flat_mag = flat_mag[flat_mag > 0]  # Remove zeros
            
            # Sample for computational efficiency
            if len(flat_mag) > 10000:
                # Deterministic seed from image content for reproducibility
                import hashlib as _hl
                _seed = int(_hl.sha256(self.ctx.image_bytes[:64]).hexdigest()[:8], 16) % (2**31)
                flat_mag = np.random.default_rng(_seed).choice(flat_mag, 10000, replace=False)
            
            # Create histogram (probability distribution)
            hist, bin_edges = np.histogram(flat_mag, bins=50, density=True)
            hist = hist / hist.sum()  # Normalize
            
            # Natural prior: power law distribution
            # For natural images, magnitude follows approximately exponential
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            natural_dist = np.exp(-bin_centers / bin_centers.mean())
            natural_dist = natural_dist / natural_dist.sum()
            
            # Add small constant to avoid log(0)
            hist = hist + 1e-10
            natural_dist = natural_dist + 1e-10
            
            # Compute KL divergence
            kl_div = entropy(hist, natural_dist)
            
            # Natural: typically < 0.8
            # AI: often > 1.2
            if kl_div > 1.0:
                score = min(1.0, (kl_div - 1.0) / 1.5)
                explanation = f"KL divergence ({kl_div:.3f}) shows distribution deviation from natural prior"
            else:
                score = 0.0
                explanation = f"KL divergence ({kl_div:.3f}) matches natural distribution"
            
            logger.info(f"KL divergence: {kl_div:.4f}")
            
            return {
                "signal_name": "KL Divergence",
                "score": float(score),
                "confidence": 0.88,
                "explanation": explanation,
                "raw_value": float(kl_div),
                "expected_range": "< 1.0",
                "method": "kl_divergence_natural_prior"
            }
            
        except Exception as e:
            logger.warning(f"KL divergence analysis failed: {e}")
            return {
                "signal_name": "KL Divergence",
                "score": 0.0,
                "confidence": 0.3,
                "explanation": "Analysis failed - insufficient data",
                "raw_value": 0.0,
                "expected_range": "N/A",
                "method": "kl_divergence_natural_prior"
            }
    

    def analyze_perturbation_stability(self) -> Dict[str, Any]:
        """
        Perturbation Stability Test
        
        Research basis: Similar to DetectGPT but for images
        
        Method:
        1. Compute original spectral slope α
        2. Add small Gaussian noise (σ=2)
        3. Recompute spectral slope
        4. Measure change: Δα = |α_original - α_perturbed|
        
        Natural images: Stable (Δα < 0.15)
        AI images: Brittle (Δα often > 0.25)
        
        Returns:
            Detection signal with score, confidence, explanation
        """
        try:
            # Compute original spectrum
            original_spectrum = self._get_radial_spectrum()
            
            # Fit power law to original
            valid_range = slice(5, min(50, len(original_spectrum)))
            log_r = np.log(np.arange(5, 5 + len(original_spectrum[valid_range])) + 1)
            
            if len(log_r) < 3:
                raise ValueError("Spectrum too short")
            
            coeffs_original = np.polyfit(log_r, original_spectrum[valid_range], 1)
            alpha_original = -coeffs_original[0]
            
            # Add small noise perturbation
            rng = np.random.default_rng(42)
            noise = rng.normal(0, 2, self.ctx.cv_gray.shape)
            perturbed_image = np.clip(self.ctx.cv_gray + noise, 0, 255).astype(np.uint8)
            
            # Compute perturbed spectrum
            f_transform = fft.fft2(perturbed_image)
            f_shift = fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            log_magnitude = np.log(magnitude + 1)
            
            center_y, center_x = self.ctx.height // 2, self.ctx.width // 2
            y, x = np.ogrid[:self.ctx.height, :self.ctx.width]
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
            
            r_max = min(center_y, center_x) // 2
            # Vectorized — same O(H*W) fix as _get_radial_spectrum.
            r_clipped = np.clip(r.ravel(), 0, r_max - 1)
            lm_flat   = log_magnitude.ravel()
            sums_p   = np.bincount(r_clipped, weights=lm_flat, minlength=r_max)
            counts_p = np.bincount(r_clipped,                   minlength=r_max).clip(1)
            perturbed_spectrum = sums_p / counts_p
            
            # Fit power law to perturbed
            perturbed_spectrum = perturbed_spectrum[:len(original_spectrum)]
            coeffs_perturbed = np.polyfit(log_r[:len(perturbed_spectrum[valid_range])], 
                                          perturbed_spectrum[valid_range], 1)
            alpha_perturbed = -coeffs_perturbed[0]
            
            # Compute stability
            delta_alpha = abs(alpha_original - alpha_perturbed)
            
            # Natural: stable (Δα < 0.2)
            # AI: brittle (Δα > 0.3)
            if delta_alpha > 0.25:
                score = min(1.0, (delta_alpha - 0.25) / 0.3)
                explanation = f"Perturbation instability (Δα={delta_alpha:.3f}) suggests synthetic generation"
            else:
                score = 0.0
                explanation = f"Perturbation stability (Δα={delta_alpha:.3f}) indicates natural origin"
            
            logger.info(f"Perturbation stability: α_orig={alpha_original:.3f}, α_pert={alpha_perturbed:.3f}, Δα={delta_alpha:.3f}")
            
            return {
                "signal_name": "Perturbation Stability",
                "score": float(score),
                "confidence": 0.85,
                "explanation": explanation,
                "raw_value": float(delta_alpha),
                "expected_range": "< 0.25",
                "method": "perturbation_stability_test"
            }
            
        except Exception as e:
            logger.warning(f"Perturbation stability analysis failed: {e}")
            return {
                "signal_name": "Perturbation Stability",
                "score": 0.0,
                "confidence": 0.3,
                "explanation": "Analysis failed - insufficient data",
                "raw_value": 0.0,
                "expected_range": "N/A",
                "method": "perturbation_stability_test"
            }
    

    def compute_all(self) -> List[Dict[str, Any]]:
        return [
            self.analyze_mahalanobis_distance(),
            self.analyze_kl_divergence(),
            self.analyze_perturbation_stability(),
        ]
