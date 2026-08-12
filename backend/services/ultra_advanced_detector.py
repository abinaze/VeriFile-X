"""
Ultra-Advanced AI Detection with Research-Validated Methods
Implements cutting-edge techniques from forensics literature.

F-16: composition-based facade, see advanced_ai_detector.py's module
docstring for the full rationale. No longer inherits AdvancedAIDetector
-- composes BasicSignals + UltraSignals instead, and re-exposes every
method from both (not just its own 3 new ones), since existing tests
call methods "inherited" from the old base class directly on instances
of this class.
"""
from typing import Dict, Any

from backend.core.logger import setup_logger
from backend.services.statistical_signals import ImageContext, BasicSignals, UltraSignals, score_and_classify

logger = setup_logger(__name__)


class UltraAdvancedDetector:
    """
    Extends AdvancedAIDetector with research-validated methods:
    1. Cross-channel noise covariance
    2. Patch-level spectral variance
    3. Natural image prior deviation
    """

    def __init__(self, image_bytes: bytes, filename: str):
        self.ctx = ImageContext.from_bytes(image_bytes, filename)

        self.image_bytes = self.ctx.image_bytes
        self.filename     = self.ctx.filename
        self.pil_image    = self.ctx.pil_image
        self.cv_image     = self.ctx.cv_image
        self.cv_gray       = self.ctx.cv_gray
        self.height, self.width = self.ctx.height, self.ctx.width

        self._basic = BasicSignals(self.ctx)
        self._ultra = UltraSignals(self.ctx)

        logger.info(f"Initialized advanced detector for {filename} ({self.width}x{self.height}px)")

    # -- inherited (from the old chain's base class) signal methods --
    def analyze_fft_radial_spectrum(self) -> Dict[str, Any]:
        return self._basic.analyze_fft_radial_spectrum()

    def analyze_dct_coefficients(self) -> Dict[str, Any]:
        return self._basic.analyze_dct_coefficients()

    def analyze_wavelet_energy(self) -> Dict[str, Any]:
        return self._basic.analyze_wavelet_energy()

    def analyze_glcm_texture(self) -> Dict[str, Any]:
        return self._basic.analyze_glcm_texture()

    def analyze_noise_residual(self) -> Dict[str, Any]:
        return self._basic.analyze_noise_residual()

    def analyze_spectral_entropy(self) -> Dict[str, Any]:
        return self._basic.analyze_spectral_entropy()

    def analyze_lbp_texture(self) -> Dict[str, Any]:
        return self._basic.analyze_lbp_texture()

    def analyze_edge_statistics(self) -> Dict[str, Any]:
        return self._basic.analyze_edge_statistics()

    def analyze_color_correlation(self) -> Dict[str, Any]:
        return self._basic.analyze_color_correlation()

    def analyze_compression_artifacts(self) -> Dict[str, Any]:
        return self._basic.analyze_compression_artifacts()

    # -- this class's own 3 signal methods --
    def analyze_rgb_noise_covariance(self) -> Dict[str, Any]:
        return self._ultra.analyze_rgb_noise_covariance()

    def analyze_patch_spectral_variance(self) -> Dict[str, Any]:
        return self._ultra.analyze_patch_spectral_variance()

    def analyze_natural_prior_deviation(self) -> Dict[str, Any]:
        return self._ultra.analyze_natural_prior_deviation()

    def detect(self) -> Dict[str, Any]:
        """Run ultra-advanced detection with all methods."""
        logger.info(f"Starting ultra-advanced detection for {self.filename}")

        signals = self._basic.compute_all() + self._ultra.compute_all()
        result = score_and_classify(signals, [(6, 1.25)], "ultra-advanced-v1.0")

        logger.info(
            f"Ultra-advanced detection complete: {result['classification']} "
            f"(p={result['ai_probability']:.3f}, "
            f"{result['suspicious_signals_count']}/{result['total_signals']} signals)"
        )

        return result
