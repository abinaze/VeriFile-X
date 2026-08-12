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

F-16: this class is now a thin, backward-compatible facade over
backend/services/statistical_signals.py's BasicSignals component,
composed rather than holding the signal logic itself -- previously the
base of a 5-level inheritance chain
(AdvancedAIDetector -> UltraAdvancedDetector -> CovarianceDetector ->
StatisticalDetector -> AdvancedEnsembleDetector). Kept because existing
tests construct this class directly and call .detect()/individual
analyze_*() methods expecting the exact original 10-signal report;
behavior is unchanged (verified via a direct before/after comparison
across multiple test images -- every signal's score/confidence and the
final ai_probability match to floating-point precision).
"""
from typing import Dict, Any, List

from backend.core.logger import setup_logger
from backend.services.statistical_signals import ImageContext, BasicSignals, score_and_classify

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
        self.ctx = ImageContext.from_bytes(image_bytes, filename)

        # Flat attributes kept for backward compatibility -- existing
        # tests/callers access these directly (e.g. detector.cv_image).
        self.image_bytes = self.ctx.image_bytes
        self.filename     = self.ctx.filename
        self.pil_image    = self.ctx.pil_image
        self.cv_image     = self.ctx.cv_image
        self.cv_gray       = self.ctx.cv_gray
        self.height, self.width = self.ctx.height, self.ctx.width

        self._signals = BasicSignals(self.ctx)

        logger.info(f"Initialized advanced detector for {filename} ({self.width}x{self.height}px)")

    def analyze_fft_radial_spectrum(self) -> Dict[str, Any]:
        return self._signals.analyze_fft_radial_spectrum()

    def analyze_dct_coefficients(self) -> Dict[str, Any]:
        return self._signals.analyze_dct_coefficients()

    def analyze_wavelet_energy(self) -> Dict[str, Any]:
        return self._signals.analyze_wavelet_energy()

    def analyze_glcm_texture(self) -> Dict[str, Any]:
        return self._signals.analyze_glcm_texture()

    def analyze_noise_residual(self) -> Dict[str, Any]:
        return self._signals.analyze_noise_residual()

    def analyze_spectral_entropy(self) -> Dict[str, Any]:
        return self._signals.analyze_spectral_entropy()

    def analyze_lbp_texture(self) -> Dict[str, Any]:
        return self._signals.analyze_lbp_texture()

    def analyze_edge_statistics(self) -> Dict[str, Any]:
        return self._signals.analyze_edge_statistics()

    def analyze_color_correlation(self) -> Dict[str, Any]:
        return self._signals.analyze_color_correlation()

    def analyze_compression_artifacts(self) -> Dict[str, Any]:
        return self._signals.analyze_compression_artifacts()

    def calculate_final_score(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensemble scoring with confidence weighting."""
        return score_and_classify(signals, [(5, 1.2)], "advanced-ai-v1.0")

    def detect(self) -> Dict[str, Any]:
        """Run complete advanced AI detection pipeline."""
        logger.info(f"Starting advanced AI detection for {self.filename}")

        signals = self._signals.compute_all()
        result = self.calculate_final_score(signals)

        logger.info(
            f"Advanced detection complete: {result['classification']} "
            f"(p={result['ai_probability']:.3f}, "
            f"{result['suspicious_signals_count']}/{result['total_signals']} signals)"
        )

        return result
