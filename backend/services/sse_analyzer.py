"""
Server-Sent Events (SSE) streaming analysis.

Runs the 30-signal forensic pipeline and streams each signal result
to the client as it completes, enabling a real-time waterfall UI.

KEY FIX: Each of the 30 signals is computed EXACTLY ONCE.
Previously the SSE path ran all 30 signals twice:
  1. Individual detector calls streamed to the UI (28 signals)
  2. generate_forensic_report() called at the end — which internally
     called AdvancedEnsembleDetector.detect() again, re-running all 30.
DIRE and OwnEmbedding were also missing from the individual events.

New architecture:
  1. Stream 19 statistical signals via the composed StatisticalDetector
     instance (detector._stat.detect()) -- F-16: AdvancedEnsembleDetector
     composes this rather than inheriting from it
  2. Stream DIRE, CLIP, OwnEmbedding individually
  3. Stream 8 forensic signals individually
  4. Call combine_signals() ONCE on the already-computed results
  5. Emit the summary event — total detector executions: 30

Protocol (each event is JSON):
  started   - analysis begun
  quality   - image quality gate result
  signal    - one signal result (name, score, confidence, explanation)
  summary   - final verdict and full report
  error     - analysis failed
"""
import json
import asyncio
from backend.core.logger import setup_logger
from backend.utils.json_safe import sanitize as _sanitize
from typing import AsyncGenerator

logger = setup_logger(__name__)


async def stream_analysis(image_bytes: bytes, filename: str) -> AsyncGenerator[str, None]:
    """
    Yield SSE-formatted strings as each detection signal completes.
    Compatible with EventSource / fetch-with-ReadableStream in the browser.
    """

    def _sse(event_type: str, data: dict) -> str:
        payload = json.dumps({"type": event_type, **data}, default=str)
        return f"data: {payload}\n\n"

    def _signal_event(sig: dict) -> str:
        return _sse("signal", {
            "signal_name": sig.get("signal_name", "unknown"),
            "score":       round(float(sig.get("score", 0.5)), 4),
            "confidence":  round(float(sig.get("confidence", 0.0)), 4),
            "explanation": sig.get("explanation", ""),
            "suspicious":  float(sig.get("score", 0.5)) > 0.5,
        })

    yield _sse("started", {
        "filename":  filename,
        "file_size": len(image_bytes),
        "message":   "Analysis started — running 30 signals",
    })

    try:
        from backend.utils.image_quality import assess_image_quality
        quality = assess_image_quality(image_bytes, filename)
        if not quality["suitable"]:
            yield _sse("error", {
                "message": f"Image unsuitable: {quality['reason']}",
                "quality": quality,
            })
            return

        yield _sse("quality", {"quality": quality})
        await asyncio.sleep(0)

        from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
        detector = AdvancedEnsembleDetector(image_bytes, filename)
        loop = asyncio.get_event_loop()

        try:
            # ── 19 statistical signals ─────────────────────────────────────
            def _run_statistical():
                # F-16: AdvancedEnsembleDetector composes a StatisticalDetector
                # instance (detector._stat) rather than inheriting from it, so
                # this calls that composed instance directly instead of the
                # old super(AdvancedEnsembleDetector, detector).detect().
                return detector._stat.detect()

            base_report = await loop.run_in_executor(None, _run_statistical)
            for sig in base_report.get("all_signals", []):
                yield _signal_event(sig)
                await asyncio.sleep(0)

            # ── DIRE (stream individually — was missing before) ────────────
            def _run_dire():
                return detector.dire_detector.detect(image_bytes, filename)

            dire_result = await loop.run_in_executor(None, _run_dire)
            yield _signal_event(dire_result)
            await asyncio.sleep(0)

            # ── CLIP ──────────────────────────────────────────────────────
            def _run_clip():
                return detector.clip_detector.detect(image_bytes, filename)

            clip_result = await loop.run_in_executor(None, _run_clip)
            yield _signal_event(clip_result)
            await asyncio.sleep(0)

            # ── OwnEmbedding (stream individually — was missing before) ───
            def _run_own():
                return detector.own_detector.detect(image_bytes, filename)

            own_result = await loop.run_in_executor(None, _run_own)
            yield _signal_event(own_result)
            await asyncio.sleep(0)

            # ── 8 forensic signals ─────────────────────────────────────────
            def _run_forensic_extras():
                from backend.services.prnu_detector       import detect_prnu
                from backend.services.ela_detector        import detect_ela
                from backend.services.metadata_forensics  import analyze_metadata
                from backend.services.dct_frequency_detector import detect_dct_artifacts
                from backend.services.jpeg_ghost_detector import detect_jpeg_ghost
                from backend.services.noise_map_detector  import detect_noise_map
                from backend.services.noiseprint_detector import detect_noiseprint
                from backend.services.cfa_detector        import detect_cfa_artifacts
                return (
                    detect_prnu(image_bytes, filename),
                    detect_ela(image_bytes, filename),
                    analyze_metadata(image_bytes, filename),
                    detect_dct_artifacts(image_bytes, filename),
                    detect_jpeg_ghost(image_bytes, filename),
                    detect_noise_map(image_bytes, filename),
                    detect_noiseprint(image_bytes, filename),
                    detect_cfa_artifacts(image_bytes, filename),
                )

            (
                prnu_result, ela_result, metadata_result, dct_result,
                jpeg_ghost_result, noise_map_result, noiseprint_result,
                cfa_result,
            ) = await loop.run_in_executor(None, _run_forensic_extras)

            for sig in (
                prnu_result, ela_result, metadata_result, dct_result,
                jpeg_ghost_result, noise_map_result, noiseprint_result,
                cfa_result,
            ):
                yield _signal_event(sig)
                await asyncio.sleep(0)

            # ── Combine all 30 pre-computed results (no re-execution) ──────
            def _combine():
                return detector.combine_signals(
                    base_report, dire_result, clip_result, own_result,
                    prnu_result, ela_result, metadata_result, dct_result,
                    jpeg_ghost_result, noise_map_result, noiseprint_result,
                    cfa_result,
                )

            ai_detection = await loop.run_in_executor(None, _combine)

            # ── Remaining report sections (no signal re-execution) ─────────
            def _build_report():
                from backend.services.image_forensics      import ImageForensics
                from backend.services.generator_attribution import attribute_generator
                from backend.services.platform_detector    import detect_platform
                from backend.services.c2pa_verifier        import verify_c2pa
                from backend.services.image_type_classifier import classify_image_type
                from backend.core.config                   import settings
                from datetime import datetime
                import uuid

                forensics   = ImageForensics(image_bytes, filename)
                exif_data   = forensics.extract_exif()
                hashes      = forensics.generate_hashes()
                tampering   = forensics.detect_tampering_indicators(exif_data)
                attribution = attribute_generator(image_bytes, filename)
                platform    = detect_platform(image_bytes, filename)
                c2pa        = verify_c2pa(image_bytes, filename)
                img_type    = classify_image_type(image_bytes, filename)
                width, height = forensics.pil_image.size

                return {
                    "evidence_id": str(uuid.uuid5(uuid.NAMESPACE_URL, hashes["sha256"])),
                    "metadata": {
                        "analysis_timestamp": datetime.now().isoformat(),
                        "analyzer_version":   settings.VERSION,
                    },
                    "file_info": {
                        "filename":        filename,
                        "format":          forensics.pil_image.format or "Unknown",
                        "mode":            forensics.pil_image.mode,
                        "width":           width,
                        "height":          height,
                        "file_size_bytes": len(image_bytes),
                    },
                    "exif_data":             exif_data,
                    "hashes":                hashes,
                    "tampering_analysis":    tampering,
                    "ai_detection":          ai_detection,
                    "generator_attribution": attribution,
                    "platform_forensics":    platform,
                    "c2pa_provenance":       c2pa,
                    "image_type":            img_type,
                    "summary": {
                        "has_metadata":                 exif_data.get("has_exif", False),
                        "suspicious_flags_count":       len(tampering["suspicious_flags"]),
                        "authenticity_confidence":      tampering["confidence"],
                        "ai_probability":               ai_detection["ai_probability"],
                        "ai_classification":            ai_detection["classification"],
                        "total_detection_signals":      ai_detection["total_signals"],
                        "suspicious_detection_signals": ai_detection["suspicious_signals_count"],
                        "predicted_generator":          attribution["predicted_generator"],
                        "platform_origin":              platform["predicted_platform"],
                        "c2pa_status":                  c2pa["provenance_status"],
                        "image_type":                   img_type["image_type"],
                    },
                }

            report = await loop.run_in_executor(None, _build_report)

        finally:
            detector.cleanup()

        # Sanitize NaN/Inf and numpy scalar types (H-5: was a local,
        # incomplete duplicate of analyze.py's sanitizer -- missing the
        # numpy-scalar conversion that copy has. Now the same shared
        # implementation both use.)
        report = _sanitize(report)

        yield _sse("summary", {
            "report":         report,
            "ai_probability": report["summary"]["ai_probability"],
            "classification": report["summary"]["ai_classification"],
            "total_signals":  report["summary"]["total_detection_signals"],
        })

    except Exception:
        logger.error("SSE analysis failed for %s", filename, exc_info=True)
        yield _sse("error", {"message": "Analysis failed. Please try again."})
