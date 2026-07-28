"""
Forensic analysis endpoints.
Security: Rate limited, validated, privacy-first.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
import hashlib
import time
from typing import List

from backend.services.image_forensics import ImageForensics
from backend.utils.validators import validate_file, FileValidationError
from backend.core.logger import setup_logger
from backend.core.cache import forensics_cache
from backend.services.metrics_collector import record_analysis
from backend.core.audit_log import log_analysis
from backend.core.config import settings

logger = setup_logger(__name__)

# Standalone limiter for this router — same config as main.py
# BUG FIX: previously no default_limits — settings.RATE_LIMIT_PER_MINUTE
# (declared in .env.example/render.yaml) had no effect anywhere. Wired
# here as the DEFAULT limit for any endpoint without its own explicit
# @limiter.limit(...) decorator — the 24 existing per-endpoint
# decorators are intentionally tuned differently per endpoint cost and
# are NOT touched by this change.
from backend.core.config import settings as _settings
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{_settings.RATE_LIMIT_PER_MINUTE}/minute"])

from backend.core.auth import require_analyst_or_demo
from fastapi import Depends

router = APIRouter(
    prefix="/api/v1/analyze",
    dependencies=[Depends(require_analyst_or_demo)],
    tags=["Forensic Analysis"]
)

# Allowed MIME types for analysis endpoint
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/heic", "image/heif"}

# Max file size for analysis (CPU-intensive operation). Previously hardcoded
# to 10MB independently of settings.MAX_ANALYSIS_SIZE_MB, so changing the
# config value had no effect on the actual enforced limit. Now derived from
# settings so there is a single source of truth.
MAX_ANALYSIS_SIZE_BYTES = settings.MAX_ANALYSIS_SIZE_MB * 1024 * 1024


@router.post(
    "/image",
    summary="Analyze image forensics",
    description="""
    Performs comprehensive forensic analysis on uploaded image.

    **Analysis includes:**
    - EXIF metadata extraction (camera, GPS, timestamps)
    - Cryptographic hash generation (SHA-256, MD5)
    - Perceptual hashing for similarity detection
    - Tampering indicator detection
    - AI-generated content detection
    - Authenticity confidence scoring

    **Privacy:** File processed in-memory, immediately discarded.

    **Rate limit:** 10 requests per minute per IP.

    **Supported formats:** JPEG, PNG, WebP

    **Caching:** Duplicate uploads return cached results instantly.
    """
)
@limiter.limit("10/minute")
async def analyze_image(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze")
):
    """
    Perform forensic analysis on uploaded image.

    Security layers:
    1. Rate limiting (10/min per IP)
    2. Content-type header check
    3. File size limit (10MB)
    4. MIME type verification via python-magic
    5. In-memory only (zero disk writes)
    6. SHA-256 hash caching (deduplication)
    """
    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            logger.warning(
                f"Rejected upload: content_type={file.content_type} "
                f"from {get_remote_address(request)}"
            )
            raise HTTPException(
                status_code=415,
                detail="Unsupported media type. Allowed: image/jpeg, image/png, image/webp"
            )

        # Check Content-Length header BEFORE reading to avoid loading huge files into RAM
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_ANALYSIS_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Payload too large. Max size: {MAX_ANALYSIS_SIZE_BYTES // (1024*1024)}MB"
                    )
            except ValueError:
                pass  # Invalid Content-Length header — proceed and check after read

        file_bytes = await file.read()

        # Apply EXIF orientation before all forensic analysis.
        # iPhone portrait photos have EXIF rotation=6 (90° CW) — without this,
        # all spatial signals (PRNU, CFA, noise map) run on a sideways image.
        try:
            from PIL import Image as _PIL_img, ImageOps as _EXIF_ops
            from io import BytesIO as _BytesIO_exif
            _img_exif = _PIL_img.open(_BytesIO_exif(file_bytes))
            _orig_fmt  = _img_exif.format or "JPEG"  # preserve before exif_transpose clears .format
            _img_exif  = _EXIF_ops.exif_transpose(_img_exif)
            _buf_exif  = _BytesIO_exif()
            _img_exif.save(_buf_exif, format=_orig_fmt)
            file_bytes = _buf_exif.getvalue()
        except Exception:
            pass  # Proceed with original bytes if EXIF correction fails

        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            logger.warning(
                f"Rejected upload: size={len(file_bytes)} bytes "
                f"exceeds {MAX_ANALYSIS_SIZE_BYTES} bytes"
            )
            raise HTTPException(
                status_code=413,
                detail=f"Payload too large. Max size: {MAX_ANALYSIS_SIZE_BYTES // (1024*1024)}MB"
            )

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        logger.info(
            f"Analyzing: file={file.filename}, "
            f"size={len(file_bytes)} bytes, "
            f"sha256={file_hash[:16]}..., "
            f"content_type={file.content_type}"
        )

        validation = validate_file(file_bytes, file.filename)

        if not validation["mime_type"].startswith("image/"):
            raise FileValidationError(
                f"File content is not an image: {validation['mime_type']}"
            )
        # Use extension_valid from validate_file (previously silently discarded)
        if not validation.get("extension_valid", True):
            raise FileValidationError(
                f"File extension not allowed: {validation.get('extension', 'unknown')}"
            )

        # Quality gate — reject images too small or unreadable for forensics
        from backend.utils.image_quality import assess_image_quality
        quality = assess_image_quality(file_bytes, file.filename)
        if not quality["suitable"]:
            raise HTTPException(
                status_code=422,
                detail=f"Image unsuitable for analysis: {quality['reason']}"
            )
        _confidence_cap = quality.get("confidence_cap", 1.0)

        cached_result = forensics_cache.get(file_hash)
        if cached_result:
            logger.info(
                f"Cache HIT: Returning cached result for {file.filename} "
                f"(saved ~2-5 seconds of processing)"
            )
            # Sanitize on cache hit too — cached entries may predate sanitizer
            import math as _math_cache
            import numpy as _np_hit
            def _sanitize_hit(obj):
                # Handle numpy scalar types explicitly (bool, int, float variants)
                if isinstance(obj, _np_hit.generic):
                    return obj.item()
                if isinstance(obj, (float, _np_hit.floating)):
                    v = float(obj)
                    return 0.0 if (_math_cache.isnan(v) or _math_cache.isinf(v)) else v
                if isinstance(obj, _np_hit.integer):
                    return int(obj)
                if isinstance(obj, dict): return {k: _sanitize_hit(v) for k, v in obj.items()}
                if isinstance(obj, list): return [_sanitize_hit(v) for v in obj]
                return obj
            return _sanitize_hit(cached_result)

        logger.info(f"Cache MISS: Running full analysis for {file.filename}")
        import asyncio as _asyncio
        _t0 = time.perf_counter()
        forensics = ImageForensics(file_bytes, file.filename)
        report = await _asyncio.get_running_loop().run_in_executor(
            None, forensics.generate_forensic_report
        )
        _latency_ms = round((time.perf_counter() - _t0) * 1000, 1)

        forensics_cache.set(file_hash, report)

        # Apply quality-based confidence cap to all signal confidences
        if _confidence_cap < 1.0:
            for sig in report.get("ai_detection", {}).get("all_signals", []):
                if "confidence" in sig:
                    sig["confidence"] = min(float(sig["confidence"]), _confidence_cap)
            logger.info("Applied confidence_cap=%.2f (image quality gate)", _confidence_cap)

        # Fire outbound webhooks (non-blocking daemon threads)
        try:
            from backend.services.webhook_manager import fire_webhooks as _fw
            _fw(
                evidence_id=report.get("evidence_id", ""),
                result=report,
                event="analysis.complete",
            )
        except Exception:
            logger.warning("Webhook fire failed", exc_info=True)

        log_analysis(
            evidence_id=report.get("evidence_id", "unknown"),
            filename=file.filename,
            file_sha256=file_hash,
            ai_probability=report["summary"]["ai_probability"],
            classification=report["summary"]["ai_classification"],
            total_signals=report["summary"]["total_detection_signals"],
            suspicious_signals=report["summary"]["suspicious_detection_signals"],
            methods_used=report["ai_detection"].get("methods_used", [])
        )

        logger.info(
            f"Analysis complete: file={file.filename}, "
            f"ai_probability={report['summary']['ai_probability']:.3f}, "
            f"classification={report['summary']['ai_classification']}"
        )

        # Sanitize any NaN/Inf float values before JSON serialization
        import math
        import numpy as _np_sanitize
        def _sanitize(obj):
            # Handle numpy scalar types (np.float64, np.int32, etc.)
            if isinstance(obj, _np_sanitize.generic):
                return obj.item()
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return 0.0
                return obj
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return obj
        report = _sanitize(report)

        # Record metrics
        try:
            record_analysis(
                ai_probability=report["summary"]["ai_probability"],
                classification=report["summary"]["ai_classification"],
                latency_ms=_latency_ms,
                predicted_generator=report["summary"].get("predicted_generator", "unknown"),
                platform_origin=report["summary"].get("platform_origin", "unknown"),
                signal_scores=report["ai_detection"].get("all_signals", []),
            )
        except Exception:
            pass

        return report

    except FileValidationError:
        logger.warning("File validation failed", exc_info=True)
        raise HTTPException(status_code=422, detail="File validation failed. Check file type and size.")

    except ValueError:
        logger.error("Value error during analysis", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid image data")

    except IOError:
        logger.error("I/O error during analysis", exc_info=True)
        raise HTTPException(status_code=422, detail="Unable to process image")

    except HTTPException:
        raise

    except Exception:
        logger.error("Unexpected analysis error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during analysis"
        )

    finally:
        await file.close()


@router.get("/history", summary="Recent analysis history")
async def get_history(limit: int = 20):
    """Return recent analysis records from audit log."""
    from backend.core.audit_log import get_recent_analyses
    return {"analyses": get_recent_analyses(limit=min(limit, 100))}


@router.get("/stats", summary="Analysis statistics")
async def get_stats():
    """Return aggregate detection statistics."""
    from backend.core.audit_log import get_stats
    return get_stats()


@router.post(
    "/image/heatmap",
    summary="Generate manipulation localization heatmap",
    description="Returns a Grad-CAM heatmap PNG (base64) highlighting suspicious regions. "
                "Submit the same image previously analyzed. Requires EfficientNet model."
)
@limiter.limit("5/minute")
async def analyze_image_heatmap(
    request: Request,
    file: UploadFile = File(..., description="Image file to generate heatmap for")
):
    """Generate Grad-CAM localization heatmap for uploaded image."""
    from backend.services.heatmap_generator import generate_heatmap

    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported media type. Allowed: image/jpeg, image/png, image/webp")

        file_bytes = await file.read()

        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")

        import asyncio as _aio_hm
        result = await _aio_hm.to_thread(generate_heatmap, file_bytes, file.filename)

        logger.info(
            f"Heatmap generated: file={file.filename}, "
            f"method={result['method']}, size={result['width']}x{result['height']}"
        )

        return {
            "filename":    file.filename,
            "heatmap_b64": result["heatmap_b64"],
            "width":       result["width"],
            "height":      result["height"],
            "method":      result["method"],
        }

    except HTTPException:
        raise
    except Exception:
        logger.error("Heatmap generation error", exc_info=True)
        raise HTTPException(status_code=500, detail="Heatmap generation failed")
    finally:
        await file.close()


@router.post(
    "/attribution",
    summary="Attribute image to AI generator",
    description="Classifies the image into: stylegan, dalle3, sd14, sdxl, midjourney, real, or unknown."
)
@limiter.limit("10/minute")
async def analyze_attribution(
    request: Request,
    file: UploadFile = File(..., description="Image file to attribute")
):
    """Attribute uploaded image to its most likely AI generator."""
    from backend.services.generator_attribution import attribute_generator

    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported media type. Allowed: image/jpeg, image/png, image/webp")

        file_bytes = await file.read()

        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")

        import asyncio as _aio_attr
        result = await _aio_attr.to_thread(attribute_generator, file_bytes, file.filename)

        logger.info(
            f"Attribution complete: file={file.filename}, "
            f"generator={result['predicted_generator']}, "
            f"confidence={result['confidence']:.3f}"
        )

        return result

    except HTTPException:
        raise
    except Exception:
        logger.error("Attribution error", exc_info=True)
        raise HTTPException(status_code=500, detail="Attribution analysis failed")
    finally:
        await file.close()


@router.post(
    "/platform",
    summary="Detect social media platform re-encoding",
    description="Identifies WhatsApp, Instagram, Discord, Telegram, Twitter/X, or Facebook compression signatures."
)
@limiter.limit("10/minute")
async def analyze_platform(
    request: Request,
    file: UploadFile = File(..., description="Image file to check platform signature")
):
    """Detect social media platform from JPEG quantization fingerprint."""
    from backend.services.platform_detector import detect_platform

    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported media type. Allowed: image/jpeg, image/png, image/webp")

        file_bytes = await file.read()

        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")

        import asyncio as _aio_plat
        result = await _aio_plat.to_thread(detect_platform, file_bytes, file.filename)

        logger.info(
            f"Platform detection: file={file.filename}, "
            f"platform={result['predicted_platform']}, "
            f"confidence={result['confidence']:.3f}"
        )

        return result

    except HTTPException:
        raise
    except Exception:
        logger.error("Platform detection error", exc_info=True)
        raise HTTPException(status_code=500, detail="Platform detection failed")
    finally:
        await file.close()

@router.post(
    "/c2pa",
    summary="Verify C2PA content credentials",
    description="Checks for C2PA provenance manifest. Returns: verified | partial | none | tampered."
)
@limiter.limit("10/minute")
async def analyze_c2pa(
    request: Request,
    file: UploadFile = File(..., description="Image file to check for C2PA credentials")
):
    """Verify C2PA content credentials in uploaded image."""
    from backend.services.c2pa_verifier import verify_c2pa

    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported media type. Allowed: image/jpeg, image/png, image/webp")
        file_bytes = await file.read()
        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")
        import asyncio as _aio_c2pa
        result = await _aio_c2pa.to_thread(verify_c2pa, file_bytes, file.filename)
        logger.info(
            f"C2PA verification: file={file.filename}, "
            f"status={result['provenance_status']}, "
            f"has_c2pa={result['has_c2pa']}"
        )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.error("C2PA verification error", exc_info=True)
        raise HTTPException(status_code=500, detail="C2PA verification failed")
    finally:
        await file.close()


@router.post(
    "/robustness",
    summary="Test adversarial robustness of detection",
    description="Runs 8 adversarial attack types at 3 intensities each. Returns per-attack robustness scores."
)
@limiter.limit("2/minute")
async def analyze_robustness(
    request: Request,
    file: UploadFile = File(..., description="Image to test robustness against")
):
    """Run adversarial robustness test suite on uploaded image."""
    from backend.services.adversarial_tester import run_robustness_test

    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported media type. Allowed: image/jpeg, image/png, image/webp")
        file_bytes = await file.read()
        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")
        import asyncio as _aio_rob
        result = await _aio_rob.to_thread(run_robustness_test, file_bytes, file.filename)
        logger.info(
            f"Robustness test: file={file.filename}, "
            f"overall={result.get('overall_robustness', 0):.3f}, "
            f"level={result.get('robustness_level', 'unknown')}"
        )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.error("Robustness test error", exc_info=True)
        raise HTTPException(status_code=500, detail="Robustness test failed")
    finally:
        await file.close()


@router.post(
    "/batch",
    summary="Batch forensic analysis — up to 10 images",
    description="Processes multiple images in one request. Returns per-image reports plus aggregate statistics, duplicate detection, risk ranking, and provenance consistency check."
)
@limiter.limit("2/minute")
async def analyze_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="Images to analyze (max 10)")
):
    """Process multiple images through the full forensic pipeline."""
    from backend.services.batch_processor import process_batch, MAX_BATCH_SIZE

    try:
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Too many files. Max {MAX_BATCH_SIZE} per batch request."
            )

        images = []
        for file in files:
            if file.content_type not in ALLOWED_IMAGE_TYPES:
                logger.warning(f"Batch: skipping {file.filename} — unsupported type {file.content_type}")
                continue
            data = await file.read()
            images.append({"filename": file.filename, "data": data})

        if not images:
            raise HTTPException(status_code=415, detail="No valid image files in batch.")

        logger.info(f"Batch analysis: {len(images)} images submitted")
        import asyncio as _aio_batch
        result = await _aio_batch.to_thread(process_batch, images)

        logger.info(
            f"Batch complete: processed={result.get('processed', 0)}, "
            f"failed={result.get('failed', 0)}, "
            f"verdict={result.get('statistics', {}).get('batch_verdict', 'unknown')}"
        )

        return result

    except HTTPException:
        raise
    except Exception:
        logger.error("Batch analysis error", exc_info=True)
        raise HTTPException(status_code=500, detail="Batch analysis failed")
    finally:
        for file in files:
            await file.close()


@router.post(
    "/segment",
    summary="Segment-level AI detection — per-tile probability grid",
)
@limiter.limit("3/minute")
async def analyze_segment(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Divide the image into overlapping 64x64 tiles and return a 2-D grid
    of per-tile AI probability scores.  Useful for detecting partial AI
    insertion (real background with AI-generated subject composited in).
    """
    try:
        image_bytes = await file.read()
        from backend.utils.validators import validate_file, FileValidationError as _FVE
        validate_file(image_bytes, file.filename or "upload")
    except _FVE as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Unexpected error validating /segment upload")
        raise HTTPException(status_code=422, detail="Invalid or unreadable image file")

    try:
        from backend.services.segment_detector import detect_segments
        result = detect_segments(image_bytes, file.filename or "upload")
        return result
    except Exception:
        logger.error("Segment detection error", exc_info=True)
        raise HTTPException(status_code=500, detail="Segment analysis failed")


@router.post(
    "/export/{fmt}",
    summary="Export forensic report in PDF, JSON, or CSV format",
    description="Re-analyze image and return report as downloadable file. fmt: pdf | json | csv"
)
@limiter.limit("5/minute")
async def export_report(
    request: Request,
    fmt: str,
    file: UploadFile = File(..., description="Image to analyze and export")
):
    """Analyze image and export forensic report in requested format."""
    from backend.services.report_exporter import export_pdf, export_json, export_csv

    fmt = fmt.lower().strip()
    if fmt not in ("pdf", "json", "csv"):
        raise HTTPException(status_code=400, detail="Format must be: pdf, json, or csv")

    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported media type. Allowed: image/jpeg, image/png, image/webp")
        file_bytes = await file.read()
        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")

        from backend.services.image_forensics import ImageForensics
        import asyncio as _aio_export
        _exp_hash = hashlib.sha256(file_bytes).hexdigest()
        report = forensics_cache.get(_exp_hash)
        if not report:
            forensics = ImageForensics(file_bytes, file.filename)
            report = await _aio_export.to_thread(forensics.generate_forensic_report)
            forensics_cache.set(_exp_hash, report)

        stem = file.filename.rsplit(".", 1)[0] if "." in file.filename else file.filename

        if fmt == "pdf":
            content      = export_pdf(report)
            media_type   = "application/pdf"
            disposition  = f'attachment; filename="{stem}_forensic_report.pdf"'
        elif fmt == "json":
            content      = export_json(report)
            media_type   = "application/json"
            disposition  = f'attachment; filename="{stem}_forensic_report.json"'
        else:  # csv
            content      = export_csv(report)
            media_type   = "text/csv"
            disposition  = f'attachment; filename="{stem}_forensic_signals.csv"'

        logger.info(f"Export: file={file.filename}, format={fmt}, size={len(content)} bytes")

        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": disposition}
        )

    except HTTPException:
        raise
    except Exception:
        logger.error("Export error", exc_info=True)
        raise HTTPException(status_code=500, detail="Export generation failed. Please try again.")
    finally:
        await file.close()


@router.post(
    "/image/stream",
    summary="Stream forensic analysis via Server-Sent Events",
    description="Returns SSE stream. Each event is JSON with type: started|quality|signal|summary|error."
)
@limiter.limit("5/minute")
async def analyze_image_stream(
    request: Request,
    file: UploadFile = File(..., description="Image to analyze")
):
    """Real-time streaming analysis — 26 signals arrive one by one as SSE events."""
    from backend.services.sse_analyzer import stream_analysis

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported: {file.content_type}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")

    filename = file.filename or "unknown"
    await file.close()

    return StreamingResponse(
        stream_analysis(file_bytes, filename),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        }
    )
