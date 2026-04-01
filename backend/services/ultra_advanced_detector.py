"""
Ultra-Advanced AI Detection with Research-Validated Methods
Implements cutting-edge techniques from forensics literature.
"""
import numpy as np
import cv2
from scipy import fft
from typing import Dict, Any

from backend.core.logger import setup_logger
from backend.services.advanced_ai_detector import AdvancedAIDetector

logger = setup_logger(__name__)


class UltraAdvancedDetector(AdvancedAIDetector):
    """
    Extends AdvancedAIDetector with research-validated methods:
    1. Cross-channel noise covariance
    2. Patch-level spectral variance
    3. Natural image prior deviation
    """
    
    def analyze_rgb_noise_covariance(self) -> Dict[str, Any]:
        """
        Cross-Channel Noise Covariance Analysis
        
        Research basis: "Under-discussed and powerful" method
        
        Real cameras: RGB channels share sensor physics → correlated noise
        AI images: Often independently synthesized → lower correlation
        """
        b, g, r = cv2.split(self.cv_image.astype(float))
        
        r_noise = cv2.Laplacian(r, cv2.CV_64F).flatten()
        g_noise = cv2.Laplacian(g, cv2.CV_64F).flatten()
        b_noise = cv2.Laplacian(b, cv2.CV_64F).flatten()
        
        noise_matrix = np.vstack([r_noise, g_noise, b_noise])
        cov_matrix = np.corrcoef(noise_matrix)
        
        rg_corr = cov_matrix[0, 1]
        rb_corr = cov_matrix[0, 2]
        gb_corr = cov_matrix[1, 2]
        
        mean_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3
        
        if mean_corr < 0.5:
            score = (0.5 - mean_corr) / 0.3
            explanation = f"RGB noise correlation ({mean_corr:.3f}) is abnormally low - channels synthesized independently"
        elif mean_corr > 0.85:
            score = (mean_corr - 0.85) / 0.15
            explanation = f"RGB noise correlation ({mean_corr:.3f}) is unnaturally high"
        else:
            score = 0.0
            explanation = f"RGB noise correlation ({mean_corr:.3f}) matches camera sensor physics"
        
        return {
            "signal_name": "RGB Noise Covariance",
            "score": float(min(1.0, score)),
            "confidence": 0.88,
            "explanation": explanation,
            "raw_value": float(mean_corr),
            "expected_range": "0.5-0.85",
            "method": "cross_channel_noise_covariance"
        }
    
    def analyze_patch_spectral_variance(self) -> Dict[str, Any]:
        """
        Patch-Level FFT Variance Analysis
        
        Research basis: "Far more robust than single-image FFT"
        """
        patch_size = 128
        alphas = []
        
        for i in range(0, self.height - patch_size, patch_size):
            for j in range(0, self.width - patch_size, patch_size):
                patch = self.cv_gray[i:i+patch_size, j:j+patch_size]
                
                f_transform = fft.fft2(patch)
                f_shift = fft.fftshift(f_transform)
                magnitude = np.abs(f_shift)
                
                center_y, center_x = patch_size // 2, patch_size // 2
                y, x = np.ogrid[:patch_size, :patch_size]
                r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
                
                r_max = patch_size // 4
                radial_profile = np.zeros(r_max)
                
                for radius in range(r_max):
                    mask = (r >= radius) & (r < radius + 1)
                    if mask.any():
                        radial_profile[radius] = magnitude[mask].mean()
                
                valid_range = slice(5, r_max - 5)
                log_r = np.log(np.arange(5, r_max - 5) + 1)
                log_power = np.log(radial_profile[valid_range] + 1e-10)
                
                if len(log_r) > 0:
                    coeffs = np.polyfit(log_r, log_power, 1)
                    alpha = -coeffs[0]
                    alphas.append(alpha)
        
        if len(alphas) < 4:
            # FIXED: Always include 'method' key
            return {
                "signal_name": "Patch Spectral Variance",
                "score": 0.0,
                "confidence": 0.3,
                "explanation": "Image too small for patch analysis",
                "raw_value": 0.0,
                "expected_range": "N/A",
                "method": "patch_level_fft_variance"  # ADDED
            }
        
        alpha_variance = np.var(alphas)
        
        if alpha_variance < 0.12:
            score = (0.12 - alpha_variance) / 0.12
            explanation = f"Spectral uniformity across patches ({alpha_variance:.4f}) suggests synthetic generation"
        else:
            score = 0.0
            explanation = f"Natural spectral variation across patches ({alpha_variance:.4f})"
        
        return {
            "signal_name": "Patch Spectral Variance",
            "score": float(min(1.0, score)),
            "confidence": 0.85,
            "explanation": explanation,
            "raw_value": float(alpha_variance),
            "expected_range": "> 0.12",
            "method": "patch_level_fft_variance"
        }
    
    def analyze_natural_prior_deviation(self) -> Dict[str, Any]:
        """
        Natural Image Prior Deviation Score
        
        Measures log-likelihood deviation from 1/f² natural prior.
        """
        f_transform = fft.fft2(self.cv_gray)
        f_shift = fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        center_y, center_x = self.height // 2, self.width // 2
        y, x = np.ogrid[:self.height, :self.width]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        r_max = min(center_y, center_x) // 2
        radial_profile = np.zeros(r_max)
        
        for radius in range(r_max):
            mask = (r >= radius) & (r < radius + 1)
            if mask.any():
                radial_profile[radius] = magnitude[mask].mean()
        
        valid_range = slice(10, r_max - 10)
        log_r = np.log(np.arange(10, r_max - 10) + 1)
        log_power = np.log(radial_profile[valid_range] + 1e-10)
        
        coeffs = np.polyfit(log_r, log_power, 1)
        alpha = -coeffs[0]
        
        deviation = abs(alpha - 2.0)
        
        if deviation > 0.4:
            score = min(1.0, deviation / 0.8)
            explanation = f"Spectral slope (α={alpha:.3f}) deviates from natural prior (α≈2.0)"
        else:
            score = 0.0
            explanation = f"Spectral slope (α={alpha:.3f}) follows natural image statistics"
        
        return {
            "signal_name": "Natural Prior Deviation",
            "score": float(score),
            "confidence": 0.80,
            "explanation": explanation,
            "raw_value": float(deviation),
            "expected_range": "< 0.4",
            "method": "natural_image_prior"
        }
    
    def detect(self) -> Dict[str, Any]:
        """
        Run ultra-advanced detection with all methods.
        
        Returns:
            Complete report with 13 detection signals
        """
        logger.info(f"Starting ultra-advanced detection for {self.filename}")
        
        base_report = super().detect()
        
        new_signals = [
            self.analyze_rgb_noise_covariance(),
            self.analyze_patch_spectral_variance(),
            self.analyze_natural_prior_deviation()
        ]
        
        all_signals = base_report["all_signals"] + new_signals
        
        total_weight = sum(s["confidence"] for s in all_signals)
        weighted_score = sum(s["score"] * s["confidence"] for s in all_signals) / total_weight
        
        suspicious_count = sum(1 for s in all_signals if s["score"] > 0.5)
        
        if suspicious_count >= 6:
            weighted_score = min(1.0, weighted_score * 1.25)
        
        if weighted_score > 0.75:
            classification = "likely_ai_generated"
            confidence = "high"
        elif weighted_score > 0.45:
            classification = "possibly_ai_generated"
            confidence = "medium"
        else:
            classification = "likely_authentic"
            confidence = "high" if weighted_score < 0.25 else "medium"
        
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
            "detection_version": "ultra-advanced-v1.0"
        }
        
        logger.info(
            f"Ultra-advanced detection complete: {classification} "
            f"(p={weighted_score:.3f}, {suspicious_count}/{len(all_signals)} signals)"
        )
        
        return result
