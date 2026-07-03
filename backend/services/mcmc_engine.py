"""
MCMC Probabilistic Authenticity Engine — Phase 24.

Replaces the single point-estimate with a probability distribution by
running a Metropolis-Hastings sampler over the signal-score space.

Why this matters
----------------
Two images can both score 0.87 yet have very different evidential quality:
  - Image A: all 30 signals agree tightly → certain, narrow interval
  - Image B: signals conflict (some 0.2, some 0.98) → uncertain, wide interval

The ensemble weighted sum cannot distinguish these cases.  MCMC draws
samples from the posterior P(authenticity | signals) and reports:
  - point_estimate  : posterior mean
  - interval_90     : 90% credible interval [5th, 95th percentile]
  - interval_50     : 50% credible interval [25th, 75th percentile]
  - std             : posterior standard deviation
  - certainty       : "high" | "medium" | "low" derived from std

Algorithm (Metropolis-Hastings with Gaussian proposal)
------------------------------------------------------
State space  : θ ∈ [0, 1]  (latent "true AI probability")
Likelihood   : product of signal likelihoods
               L(signal_i | θ) = N(score_i; θ, σ_i²)
               where σ_i = (1 - confidence_i) * 0.25 + 0.05
Prior        : Beta(2, 2) — weakly regularising toward 0.5
Proposal     : θ' = θ + N(0, 0.05²), reflected at [0,1]
Chain        : 2000 samples, 500 burn-in, thinning=2
"""

import math
import numpy as np
from typing import Any, Dict, List

from backend.core.logger import setup_logger

logger = setup_logger(__name__)

# MCMC hyperparameters
_N_SAMPLES     = 2000
_BURN_IN       = 500
_PROPOSAL_STD  = 0.05
_THINNING      = 2
_PRIOR_ALPHA   = 2.0   # Beta prior α
_PRIOR_BETA    = 2.0   # Beta prior β


def _log_prior(theta: float) -> float:
    """Log Beta(α, β) prior, returns -inf outside [0,1]."""
    if theta <= 0.0 or theta >= 1.0:
        return -math.inf
    return (
        (_PRIOR_ALPHA - 1.0) * math.log(theta)
        + (_PRIOR_BETA  - 1.0) * math.log(1.0 - theta)
    )


def _log_likelihood(theta: float, scores: np.ndarray, sigmas: np.ndarray) -> float:
    """
    Log-likelihood: each signal score is modelled as
    N(score_i | theta, sigma_i²).
    """
    residuals = scores - theta
    return float(-0.5 * np.sum((residuals / sigmas) ** 2 + np.log(2 * math.pi * sigmas ** 2)))


def _log_posterior(theta: float, scores: np.ndarray, sigmas: np.ndarray) -> float:
    lp = _log_prior(theta)
    if math.isinf(lp):
        return -math.inf
    return lp + _log_likelihood(theta, scores, sigmas)


def run_mcmc(
    signals: List[Dict[str, Any]],
    point_estimate: float,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """
    Run Metropolis-Hastings over the signal space and return a posterior
    probability distribution dict.

    Args:
        signals:        List of signal dicts with 'score' and 'confidence'.
        point_estimate: Ensemble weighted sum (used as chain start point).
        rng_seed:       Seed for reproducibility.

    Returns:
        {
          "point_estimate": float,
          "interval_90":    [float, float],
          "interval_50":    [float, float],
          "std":            float,
          "certainty":      "high"|"medium"|"low",
          "n_samples":      int,
          "acceptance_rate": float,
        }
    """
    # Filter to signals that have meaningful confidence
    usable = [s for s in signals if s.get("confidence", 0.0) > 0.0]

    if not usable:
        return _fallback(point_estimate, reason="no signals with confidence > 0")

    scores = np.array([s["score"] for s in usable], dtype=np.float64)
    confs  = np.array([s.get("confidence", 0.5) for s in usable], dtype=np.float64)
    # Signal uncertainty: high-confidence signal = tight sigma, low = loose
    sigmas = (1.0 - confs) * 0.25 + 0.05
    sigmas = np.clip(sigmas, 0.02, 0.40)

    rng        = np.random.default_rng(rng_seed)
    theta      = float(np.clip(point_estimate, 0.01, 0.99))
    log_post   = _log_posterior(theta, scores, sigmas)

    chain      : List[float] = []
    accepted   = 0
    total      = 0

    total_steps = _BURN_IN + _N_SAMPLES * _THINNING

    for step in range(total_steps):
        # Gaussian proposal reflected at boundaries
        proposal = theta + rng.normal(0.0, _PROPOSAL_STD)
        proposal = float(np.clip(proposal, 1e-6, 1.0 - 1e-6))

        log_post_prop = _log_posterior(proposal, scores, sigmas)
        log_alpha     = log_post_prop - log_post

        if math.log(max(rng.uniform(0.0, 1.0), 1e-300)) < log_alpha:
            theta    = proposal
            log_post = log_post_prop
            accepted += 1
        total += 1

        if step >= _BURN_IN and (step - _BURN_IN) % _THINNING == 0:
            chain.append(theta)

    chain_arr      = np.array(chain, dtype=np.float64)
    acceptance_rate = accepted / total

    posterior_mean = float(np.mean(chain_arr))
    posterior_std  = float(np.std(chain_arr))
    p5, p25, p75, p95 = np.percentile(chain_arr, [5, 25, 75, 95]).tolist()

    # Certainty label derived from posterior std
    if posterior_std < 0.05:
        certainty = "high"
    elif posterior_std < 0.12:
        certainty = "medium"
    else:
        certainty = "low"

    logger.info(
        "MCMC: n_signals=%d mean=%.3f std=%.3f CI90=[%.3f,%.3f] "
        "certainty=%s acceptance=%.2f",
        len(usable), posterior_mean, posterior_std,
        p5, p95, certainty, acceptance_rate,
    )

    return {
        "point_estimate":  round(posterior_mean, 4),
        "interval_90":     [round(p5, 4),  round(p95, 4)],
        "interval_50":     [round(p25, 4), round(p75, 4)],
        "std":             round(posterior_std, 4),
        "certainty":       certainty,
        "n_samples":       len(chain),
        "acceptance_rate": round(acceptance_rate, 3),
    }


def _fallback(point_estimate: float, reason: str) -> Dict[str, Any]:
    logger.warning("MCMC fallback: %s", reason)
    half_w = 0.15
    return {
        "point_estimate": round(point_estimate, 4),
        "interval_90":    [round(max(0.0, point_estimate - half_w * 2), 4),
                           round(min(1.0, point_estimate + half_w * 2), 4)],
        "interval_50":    [round(max(0.0, point_estimate - half_w), 4),
                           round(min(1.0, point_estimate + half_w), 4)],
        "std":            0.10,
        "certainty":      "low",
        "n_samples":      0,
        "acceptance_rate": 0.0,
    }
