"""
Server-Sent Events (SSE) streaming analysis.

Runs the 26-signal forensic pipeline and streams each signal result
to the client as it completes, enabling a real-time waterfall UI.

Protocol:
  Each event is JSON with type:
    "started"   - analysis begun, metadata
    "signal"    - one signal result (name, score, confidence)
    "summary"   - final verdict and full report
    "error"     - analysis failed
"""
import json
import asyncio
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


async def stream_analysis(image_bytes: bytes, filename: str) -> AsyncGenerator[str, None]:
    """
    Yield SSE-formatted events as each detection signal completes.
    Compatible with EventSource in the browser.
    """

    def _sse(event_type: str, data: dict) -> str:
        payload = json.dumps({"type": event_type, **data}, default=str)
        return f"data: {payload}\n\n"

    yield _sse("started", {
        "filename": filename,
        "file_size": len(image_bytes),
        "message": "Analysis started — running 26 signals",
    })

    try:
        from backend.utils.image_quality import assess_image_quality

        # Quality gate
        quality = assess_image_quality(image_bytes, filename)
        if not quality["suitable"]:
            yield _sse("error", {
                "message": f"Image unsuitable: {quality['reason']}",
                "quality": quality,
            })
            return

        yield _sse("quality", {"quality": quality})
        await asyncio.sleep(0)

        # ── Statistical signals (run as blocking in thread) ──────────────────
        loop = asyncio.get_event_loop()

        def _run_statistical():
            from backend.services.statistical_detector import StatisticalDetector
            det = StatisticalDetector(image_bytes, filename)
            return det.detect()

        stat_report = await loop.run_in_executor(None, _run_statistical)
        for sig in stat_report.get("all_signals", []):
            yield _sse("signal", {
                "signal_name": sig["signal_name"],
                "score":       round(sig["score"], 4),
                "confidence":  round(sig["confidence"], 4),
                "explanation": sig.get("explanation", ""),
                "suspicious":  sig["score"] > 0.5,
            })
            await asyncio.sleep(0)

        # ── Deep learning signals ────────────────────────────────────────────
        def _run_extra():
            from backend.services.prnu_detector import detect_prnu
            from backend.services.ela_detector import detect_ela
            from backend.services.metadata_forensics import analyze_metadata
            from backend.services.dct_frequency_detector import detect_dct_artifacts
            return [
                detect_prnu(image_bytes, filename),
                detect_ela(image_bytes, filename),
                analyze_metadata(image_bytes, filename),
                detect_dct_artifacts(image_bytes, filename),
            ]

        extra_signals = await loop.run_in_executor(None, _run_extra)
        for sig in extra_signals:
            yield _sse("signal", {
                "signal_name": sig["signal_name"],
                "score":       round(sig["score"], 4),
                "confidence":  round(sig["confidence"], 4),
                "explanation": sig.get("explanation", ""),
                "suspicious":  sig["score"] > 0.5,
            })
            await asyncio.sleep(0)

        # ── CLIP signal ──────────────────────────────────────────────────────
        def _run_clip():
            from backend.services.clip_detector import CLIPDetector
            det = CLIPDetector()
            return det.detect(image_bytes, filename)

        clip_sig = await loop.run_in_executor(None, _run_clip)
        yield _sse("signal", {
            "signal_name": clip_sig["signal_name"],
            "score":       round(clip_sig["score"], 4),
            "confidence":  round(clip_sig["confidence"], 4),
            "explanation": clip_sig.get("explanation", ""),
            "suspicious":  clip_sig["score"] > 0.5,
        })
        await asyncio.sleep(0)

        # ── Full report for final summary ────────────────────────────────────
        def _full_report():
            from backend.services.image_forensics import ImageForensics
            return ImageForensics(image_bytes, filename).generate_forensic_report()

        report = await loop.run_in_executor(None, _full_report)

        import math
        def _sanitize(obj):
            if isinstance(obj, float):
                return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
            if isinstance(obj, dict): return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_sanitize(v) for v in obj]
            return obj
        report = _sanitize(report)

        yield _sse("summary", {
            "report":          report,
            "ai_probability":  report["summary"]["ai_probability"],
            "classification":  report["summary"]["ai_classification"],
            "total_signals":   report["summary"]["total_detection_signals"],
        })

    except Exception:
        logger.error("SSE analysis failed for %s", filename, exc_info=True)
        yield _sse("error", {"message": "Analysis failed. Please try again."})
