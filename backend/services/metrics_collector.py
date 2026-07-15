"""
Metrics Collector for VeriFile-X.

Tracks real-time system metrics without external dependencies:
  - Request counts and rates
  - Detection score distributions
  - Classification breakdowns
  - Error rates
  - Response times
  - Signal-level statistics

Thread-safe in-memory store. Resets on restart.
Designed for use with /api/v1/metrics endpoint and future
Prometheus/Grafana integration.
"""
import time
from backend.core.logger import setup_logger
import threading
from typing import Dict, Any, List
from collections import deque, defaultdict
from datetime import datetime, timezone

logger = setup_logger(__name__)

_lock = threading.Lock()

# Rolling window of last 1000 requests
_WINDOW = 1000

_requests: deque   = deque(maxlen=_WINDOW)
_errors: deque     = deque(maxlen=_WINDOW)
_scores: deque     = deque(maxlen=_WINDOW)
_latencies: deque  = deque(maxlen=_WINDOW)
_classes: dict     = defaultdict(int)
_generators: dict  = defaultdict(int)
_platforms: dict   = defaultdict(int)
_signals: dict     = defaultdict(list)  # signal_name -> list of scores
_start_time        = time.time()


def record_analysis(
    ai_probability: float,
    classification: str,
    latency_ms: float,
    predicted_generator: str = "unknown",
    platform_origin: str = "unknown",
    signal_scores: List[Dict[str, Any]] = None,
    error: bool = False,
) -> None:
    """Record a completed analysis for metrics tracking."""
    now = time.time()
    with _lock:
        _requests.append(now)
        if error:
            _errors.append(now)
        else:
            _scores.append(ai_probability)
            _latencies.append(latency_ms)
            _classes[classification] += 1
            _generators[predicted_generator] += 1
            _platforms[platform_origin] += 1
            if signal_scores:
                for sig in signal_scores:
                    name  = sig.get("signal_name", "unknown")
                    score = sig.get("score", 0.5)
                    _signals[name].append(score)
                    # Keep only last 100 per signal
                    if len(_signals[name]) > 100:
                        _signals[name] = _signals[name][-100:]


def get_metrics() -> Dict[str, Any]:
    """Return current metrics snapshot."""
    now = time.time()
    uptime = now - _start_time

    with _lock:
        total_requests  = len(_requests)
        total_errors    = len(_errors)
        scores_list     = list(_scores)
        latencies_list  = list(_latencies)

        # Requests in last 60s
        recent_req  = sum(1 for t in _requests if now - t < 60)
        recent_err  = sum(1 for t in _errors   if now - t < 60)

        # Score distribution
        if scores_list:
            import statistics
            score_mean   = round(statistics.mean(scores_list), 4)
            score_median = round(statistics.median(scores_list), 4)
            score_stdev  = round(statistics.stdev(scores_list), 4) if len(scores_list) > 1 else 0.0
            ai_rate      = round(sum(1 for s in scores_list if s > 0.5) / len(scores_list), 4)
        else:
            score_mean = score_median = score_stdev = ai_rate = 0.0

        # Latency stats
        if latencies_list:
            import statistics
            lat_mean = round(statistics.mean(latencies_list), 1)
            lat_p95  = round(sorted(latencies_list)[int(len(latencies_list) * 0.95)], 1)
        else:
            lat_mean = lat_p95 = 0.0

        # Signal average scores
        signal_stats = {
            name: {
                "mean_score": round(sum(vals)/len(vals), 4),
                "n_samples":  len(vals),
            }
            for name, vals in _signals.items()
            if vals
        }

        return {
            "uptime_seconds":          round(uptime),
            "uptime_human":            _format_uptime(uptime),
            "timestamp":               datetime.now(timezone.utc).isoformat(),
            "requests": {
                "total_in_window":     total_requests,
                "errors_in_window":    total_errors,
                "requests_last_60s":   recent_req,
                "errors_last_60s":     recent_err,
                "error_rate":          round(total_errors / max(total_requests, 1), 4),
            },
            "detection": {
                "mean_ai_probability": score_mean,
                "median_ai_probability": score_median,
                "stdev_ai_probability": score_stdev,
                "ai_positive_rate":    ai_rate,
                "n_scored":            len(scores_list),
            },
            "performance": {
                "mean_latency_ms":     lat_mean,
                "p95_latency_ms":      lat_p95,
                "n_timed":             len(latencies_list),
            },
            "classification_breakdown": dict(_classes),
            "generator_breakdown":     dict(_generators),
            "platform_breakdown":      dict(_platforms),
            "signal_statistics":       signal_stats,
        }


def reset_metrics() -> None:
    """Reset all metrics (admin use only)."""
    global _start_time
    with _lock:
        _requests.clear()
        _errors.clear()
        _scores.clear()
        _latencies.clear()
        _classes.clear()
        _generators.clear()
        _platforms.clear()
        _signals.clear()
        _start_time = time.time()
    logger.info("Metrics reset")


def _format_uptime(seconds: float) -> str:
    days    = int(seconds // 86400)
    hours   = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"
