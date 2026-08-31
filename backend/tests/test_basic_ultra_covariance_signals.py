"""
Tests for individual BasicSignals/UltraSignals/CovarianceSignals methods,
exercised via StatisticalDetector -- the real, composition-based class these
signal implementations actually run through in production.

H-5, part C (audit finding): advanced_ai_detector.py, ultra_advanced_detector.py,
and covariance_detector.py were confirmed to have zero production imports
anywhere in the codebase -- they were kept alive only by their own dedicated
test files, which directly constructed them. This file replaces those three
test files (test_advanced_ai_detector.py, test_ultra_advanced_detector.py,
test_covariance_detector.py, 18 tests / 232 lines total) after auditing every
assertion in them against the real, currently-in-production test suite:

  - Every "complete detection" test (e.g. asserting exactly 10, 13, or 16
    signals from calling .detect() on one of the dead facades) tested a scoped
    subset of signals that no longer exists anywhere reachable -- StatisticalDetector
    always runs the full 19. Not preserved; nothing production-relevant to
    preserve it against.
  - Every "forensics_integration" test (asserting total_detection_signals == 30
    via ImageForensics) was a near-identical duplicate across all three dead
    files, and is already covered independently and repeatedly by
    test_determinism.py, test_invariants.py, and test_api_stress.py. Not
    preserved.
  - test_fft_radial_spectrum (from the AdvancedAIDetector file) overlaps with
    test_signal_quality.py's existing test_fft_radial_output_range and
    test_radial_slope_in_expected_range. Not preserved as a separate test.
  - Every remaining assertion -- specific per-signal signal_name/method/
    confidence/raw_value checks for DCT Coefficients, Wavelet Energy, RGB
    Noise Covariance, Patch Spectral Variance, Natural Prior Deviation,
    Eigenvalue Spread, Local Covariance Consistency, and Patch Anisotropy
    Variance -- was NOT already covered elsewhere and is preserved below,
    rewritten against StatisticalDetector (which exposes every one of these
    methods directly; see that class's own docstring for why).

Net test count goes from 18 (many redundant or testing now-unreachable
facade-only behavior) to 8 (every one testing something real and otherwise
uncovered). This is a reduction in count, not in coverage of anything that
still exists to be covered.
"""
from backend.services.statistical_detector import StatisticalDetector


def test_dct_coefficients_signal_shape(sample_image_bytes):
    """Preserved from test_advanced_ai_detector.py::test_dct_coefficients."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_dct_coefficients()

    assert result["signal_name"] == "DCT Coefficients"
    assert 0 <= result["score"] <= 1


def test_wavelet_energy_signal_shape(sample_image_bytes):
    """Preserved from test_advanced_ai_detector.py::test_wavelet_energy."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_wavelet_energy()

    assert result["signal_name"] == "Wavelet Energy"
    assert "raw_value" in result


def test_rgb_noise_covariance_signal_shape(sample_image_bytes):
    """Preserved from test_ultra_advanced_detector.py::test_rgb_noise_covariance."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_rgb_noise_covariance()

    assert result["signal_name"] == "RGB Noise Covariance"
    assert 0 <= result["score"] <= 1
    assert result["confidence"] == 0.88
    assert "raw_value" in result
    assert result["method"] == "cross_channel_noise_covariance"


def test_patch_spectral_variance_signal_shape(sample_image_bytes):
    """Preserved from test_ultra_advanced_detector.py::test_patch_spectral_variance."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_patch_spectral_variance()

    assert result["signal_name"] == "Patch Spectral Variance"
    assert 0 <= result["score"] <= 1
    assert "explanation" in result


def test_natural_prior_deviation_signal_shape(sample_image_bytes):
    """Preserved from test_ultra_advanced_detector.py::test_natural_prior_deviation."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_natural_prior_deviation()

    assert result["signal_name"] == "Natural Prior Deviation"
    assert 0 <= result["score"] <= 1
    assert result["confidence"] == 0.80
    assert result["method"] == "natural_image_prior"


def test_eigenvalue_spread_signal_shape(sample_image_bytes):
    """Preserved from test_covariance_detector.py::test_eigenvalue_spread and
    ::test_eigenvalue_spread_mathematical_properties. The raw_value>0 and
    expected_range-present parts of those two tests are already covered by
    test_signal_quality.py's test_eigenvalue_spread_bounds and
    test_all_signals_have_required_fields respectively -- only the specific
    signal_name/confidence/method values are preserved here."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_eigenvalue_spread()

    assert result["signal_name"] == "Eigenvalue Spread"
    assert result["confidence"] == 0.90
    assert result["method"] == "covariance_eigenvalue_analysis"


def test_local_covariance_consistency_signal_shape(sample_image_bytes):
    """Preserved from test_covariance_detector.py::test_local_covariance_consistency."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_local_covariance_consistency()

    assert result["signal_name"] == "Local Covariance Consistency"
    assert 0 <= result["score"] <= 1
    assert result["confidence"] >= 0.3  # May be low for small images
    assert result["method"] == "local_covariance_consistency"


def test_patch_anisotropy_variance_signal_shape(sample_image_bytes):
    """Preserved from test_covariance_detector.py::test_patch_anisotropy_variance."""
    detector = StatisticalDetector(sample_image_bytes, "test.png")
    result = detector.analyze_patch_anisotropy_variance()

    assert result["signal_name"] == "Patch Anisotropy Variance"
    assert 0 <= result["score"] <= 1
    assert result["confidence"] >= 0.3
    assert result["method"] == "patch_anisotropy_variance"
