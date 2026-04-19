"""
Statistical Modeling & Probability Analysis for AI Detection
Implements cutting-edge probability-based methods from research.
"""
import numpy as np
from typing import Dict, Any, Tuple
from scipy import fft
from scipy.stats import entropy

from backend.core.logger import setup_logger
from backend.services.covariance_detector import CovarianceDetector

logger = setup_logger(__name__)


class StatisticalDetector(CovarianceDetector):
    """
    Extends CovarianceDetector with statistical modeling methods.
    
    New methods:
    1. Mahalanobis distance in frequency space
    2. KL divergence from natural image prior
    3. Perturbation stability testing
    """
    
    # Natural image frequency model (precomputed from research)
    # These are approximate values from natural image statistics literature
    _natural_model_cache: dict = {}  # keyed by r_max for size-safety
    _natural_model_lock = None  # class-level threading.Lock
    
    def _get_radial_spectrum(self) -> np.ndarray:
        """
        Extract radial power spectrum from image.
        
        Returns:
            1D array of radial spectrum values
        """
        # Compute FFT
        f_transform = fft.fft2(self.cv_gray)
        f_shift = fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Compute radial profile
        center_y, center_x = self.height // 2, self.width // 2
        y, x = np.ogrid[:self.height, :self.width]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        # Take log of magnitude
        log_magnitude = np.log(magnitude + 1)
        
        # Radial average
        r_max = min(center_y, center_x) // 2
        radial_profile = np.zeros(r_max)
        
        for radius in range(r_max):
            mask = (r >= radius) & (r < radius + 1)
            if mask.any():
                radial_profile[radius] = log_magnitude[mask].mean()
        
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
        r_max = min(self.height, self.width) // 4
        
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
            import threading as _thr
            if StatisticalDetector._natural_model_lock is None:
                StatisticalDetector._natural_model_lock = _thr.Lock()
            r_key = min(self.height, self.width) // 4
            with StatisticalDetector._natural_model_lock:
                if r_key not in StatisticalDetector._natural_model_cache:
                    StatisticalDetector._natural_model_cache[r_key] =                         self._build_natural_model()
                natural_mean, natural_cov =                     StatisticalDetector._natural_model_cache[r_key]
            
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
            # Get magnitude spectrum
            f_transform = fft.fft2(self.cv_gray)
            magnitude = np.abs(f_transform)
            
            # Flatten and normalize to probability distribution
            flat_mag = magnitude.flatten()
            flat_mag = flat_mag[flat_mag > 0]  # Remove zeros
            
            # Sample for computational efficiency
            if len(flat_mag) > 10000:
                # Deterministic seed from image content for reproducibility
                import hashlib as _hl
                _seed = int(_hl.sha256(self.image_bytes[:64]).hexdigest()[:8], 16) % (2**31)
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
            np.random.seed(42)
            noise = np.random.normal(0, 2, self.cv_gray.shape)
            perturbed_image = np.clip(self.cv_gray + noise, 0, 255).astype(np.uint8)
            
            # Compute perturbed spectrum
            f_transform = fft.fft2(perturbed_image)
            f_shift = fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            log_magnitude = np.log(magnitude + 1)
            
            center_y, center_x = self.height // 2, self.width // 2
            y, x = np.ogrid[:self.height, :self.width]
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
            
            r_max = min(center_y, center_x) // 2
            perturbed_spectrum = np.zeros(r_max)
            
            for radius in range(r_max):
                mask = (r >= radius) & (r < radius + 1)
                if mask.any():
                    perturbed_spectrum[radius] = log_magnitude[mask].mean()
            
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
    
    def detect(self) -> Dict[str, Any]:
        """
        Run complete statistical detection with all methods.
        
        Returns:
            Complete report with 19 detection signals
        """
        logger.info(f"Starting statistical detection for {self.filename}")
        
        # Run parent class methods (16 signals)
        base_report = super().detect()
        
        # Add new statistical methods (3 signals)
        new_signals = [
            self.analyze_mahalanobis_distance(),
            self.analyze_kl_divergence(),
            self.analyze_perturbation_stability()
        ]
        
        # Combine all signals
        all_signals = base_report["all_signals"] + new_signals
        
        # Recalculate final score with all 19 signals
        total_weight = sum(s["confidence"] for s in all_signals)
        weighted_score = sum(s["score"] * s["confidence"] for s in all_signals) / total_weight
        
        suspicious_count = sum(1 for s in all_signals if s["score"] > 0.5)
        
        # Boost if multiple signals agree (more aggressive with 19 signals)
        if suspicious_count >= 10:
            weighted_score = min(1.0, weighted_score * 1.35)
        elif suspicious_count >= 8:
            weighted_score = min(1.0, weighted_score * 1.25)
        elif suspicious_count >= 6:
            weighted_score = min(1.0, weighted_score * 1.15)
        
        # Classification
        if weighted_score > 0.75:
            classification = "likely_ai_generated"
            confidence = "high"
        elif weighted_score > 0.45:
            classification = "possibly_ai_generated"
            confidence = "medium"
        else:
            classification = "likely_authentic"
            confidence = "high" if weighted_score < 0.25 else "medium"
        
        # Top reasons
        sorted_signals = sorted(all_signals, key=lambda x: x["score"], reverse=True)
        top_reasons = [s["explanation"] for s in sorted_signals[:3]]
        
        result = {
            "ai_probability": float(weighted_score),
            "classification": classification,
            "confidence": confidence,
            "suspicious_signals_count": suspicious_count,
            "total_signals": len(all_signals),
            "all_signals": all_signals,
            "top_reasons": top_reasons,
            "summary": f"Analyzed using {len(all_signals)} independent signals. "
                      f"{suspicious_count} signals indicate AI generation.",
            "detection_version": "statistical-modeling-v1.0"
        }
        
        logger.info(
            f"Statistical detection complete: {classification} "
            f"(p={weighted_score:.3f}, {suspicious_count}/{len(all_signals)} signals)"
        )
        
        return result
