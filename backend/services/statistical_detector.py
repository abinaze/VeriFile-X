"""
Statistical Modeling & Probability Analysis for AI Detection
Implements cutting-edge probability-based methods from research.

F-16: composition-based facade, see advanced_ai_detector.py's module
docstring for the full rationale. Composes BasicSignals + UltraSignals
+ CovarianceSignals + AdvancedSignals (all 19 signals) and re-exposes
every method from all four -- test_signal_quality.py in particular
calls methods from every "level" of the old inheritance chain directly
on a StatisticalDetector instance (e.g. .analyze_fft_radial_spectrum(),
originally defined 4 levels down on the base class), so this facade
must expose the full set, not just its own 3 new methods.
"""
from typing import Dict, Any

from backend.core.logger import setup_logger
from backend.services.statistical_signals import (
    ImageContext, BasicSignals, UltraSignals, CovarianceSignals, AdvancedSignals, score_and_classify
)

logger = setup_logger(__name__)


class StatisticalDetector:
    """
    Extends CovarianceDetector with statistical modeling methods.

    New methods:
    1. Mahalanobis distance in frequency space
    2. KL divergence from natural image prior
    3. Perturbation stability testing
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
        self._cov   = CovarianceSignals(self.ctx)
        self._adv   = AdvancedSignals(self.ctx)

        logger.info(f"Initialized advanced detector for {filename} ({self.width}x{self.height}px)")

    # -- inherited signal methods (base + ultra + covariance levels) --
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

    def analyze_rgb_noise_covariance(self) -> Dict[str, Any]:
        return self._ultra.analyze_rgb_noise_covariance()

    def analyze_patch_spectral_variance(self) -> Dict[str, Any]:
        return self._ultra.analyze_patch_spectral_variance()

    def analyze_natural_prior_deviation(self) -> Dict[str, Any]:
        return self._ultra.analyze_natural_prior_deviation()

    def analyze_eigenvalue_spread(self) -> Dict[str, Any]:
        return self._cov.analyze_eigenvalue_spread()

    def analyze_local_covariance_consistency(self) -> Dict[str, Any]:
        return self._cov.analyze_local_covariance_consistency()

    def analyze_patch_anisotropy_variance(self) -> Dict[str, Any]:
        return self._cov.analyze_patch_anisotropy_variance()

    # -- this class's own 3 signal methods --
    def analyze_mahalanobis_distance(self) -> Dict[str, Any]:
        return self._adv.analyze_mahalanobis_distance()

    def analyze_kl_divergence(self) -> Dict[str, Any]:
        return self._adv.analyze_kl_divergence()

    def analyze_perturbation_stability(self) -> Dict[str, Any]:
        return self._adv.analyze_perturbation_stability()

    def detect(self) -> Dict[str, Any]:
        """
        Run complete statistical detection with all methods.

        Returns:
            Complete report with 19 detection signals
        """
        logger.info(f"Starting statistical detection for {self.filename}")

        signals = (
            self._basic.compute_all() + self._ultra.compute_all()
            + self._cov.compute_all() + self._adv.compute_all()
        )
        result = score_and_classify(signals, [(10, 1.35), (8, 1.25), (6, 1.15)], "statistical-modeling-v1.0")

        logger.info(
            f"Statistical detection complete: {result['classification']} "
            f"(p={result['ai_probability']:.3f}, "
            f"{result['suspicious_signals_count']}/{result['total_signals']} signals)"
        )

        return result
