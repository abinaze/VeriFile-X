"""
Advanced Covariance & Eigenvalue Analysis for AI Detection
Implements cutting-edge forensic methods based on sensor physics.
"""
import numpy as np
import cv2
from typing import Dict, Any
from scipy.linalg import eig

from backend.core.logger import setup_logger
from backend.services.ultra_advanced_detector import UltraAdvancedDetector

logger = setup_logger(__name__)


class CovarianceDetector(UltraAdvancedDetector):
    """
    Extends UltraAdvancedDetector with eigenvalue and local covariance analysis.
    
    New methods:
    1. Covariance eigenvalue spread analysis
    2. Local covariance consistency across patches
    3. Patch-level anisotropy variance
    """
    
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
        b, g, r = cv2.split(self.cv_image.astype(float))
        
        # Extract high-frequency noise using Laplacian
        r_noise = cv2.Laplacian(r, cv2.CV_64F).flatten()
        g_noise = cv2.Laplacian(g, cv2.CV_64F).flatten()
        b_noise = cv2.Laplacian(b, cv2.CV_64F).flatten()
        
        # Sample for computational efficiency (use 10k pixels max)
        if len(r_noise) > 10000:
            # Use content-derived seed for determinism — same image always
            # produces the same sample even under concurrent requests.
            import hashlib as _hl
            _seed = int(_hl.sha256(self.image_bytes[:64]).hexdigest()[:8], 16) % (2**31)
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
        for i in range(0, self.height - patch_size, patch_size):
            for j in range(0, self.width - patch_size, patch_size):
                patch = self.cv_image[i:i+patch_size, j:j+patch_size]
                
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
        for i in range(0, self.height - patch_size, patch_size):
            for j in range(0, self.width - patch_size, patch_size):
                patch = self.cv_gray[i:i+patch_size, j:j+patch_size].astype(float)
                
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
    
    def detect(self) -> Dict[str, Any]:
        """
        Run complete covariance detection with all methods.
        
        Returns:
            Complete report with 16 detection signals
        """
        logger.info(f"Starting covariance detection for {self.filename}")
        
        # Run parent class methods (13 signals)
        base_report = super().detect()
        
        # Add new covariance methods (3 signals)
        new_signals = [
            self.analyze_eigenvalue_spread(),
            self.analyze_local_covariance_consistency(),
            self.analyze_patch_anisotropy_variance()
        ]
        
        # Combine all signals
        all_signals = base_report["all_signals"] + new_signals
        
        # Recalculate final score with all 16 signals
        total_weight = sum(s["confidence"] for s in all_signals)
        weighted_score = sum(s["score"] * s["confidence"] for s in all_signals) / total_weight
        
        suspicious_count = sum(1 for s in all_signals if s["score"] > 0.5)
        
        # Boost if multiple signals agree (more aggressive with 16 signals)
        if suspicious_count >= 8:
            weighted_score = min(1.0, weighted_score * 1.3)
        elif suspicious_count >= 6:
            weighted_score = min(1.0, weighted_score * 1.2)
        
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
            "detection_version": "covariance-advanced-v1.0"
        }
        
        logger.info(
            f"Covariance detection complete: {classification} "
            f"(p={weighted_score:.3f}, {suspicious_count}/{len(all_signals)} signals)"
        )
        
        return result
