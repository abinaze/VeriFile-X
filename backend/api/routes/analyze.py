"""
Forensic analysis endpoints.
Security: Rate limited, validated, privacy-first.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
import hashlib
from typing import List

from backend.services.image_forensics import ImageForensics
from backend.utils.validators import validate_file, FileValidationError
from backend.core.logger import setup_logger
from backend.core.cache import forensics_cache
from backend.core.audit_log import log_analysis

# In-memory heatmap store keyed by evidence_id
_heatmap_store: dict = {}

logger = setup_logger(__name__)

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(
    prefix="/api/v1/analyze",
    tags=["Forensic Analysis"]
)

# Allowed MIME types for analysis endpoint
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Max file size: 10MB for analysis (CPU-intensive operation)
MAX_ANALYSIS_SIZE_BYTES = 10 * 1024 * 1024


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
                detail=f"Unsupported media type: {file.content_type}. "
                       f"Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}"
            )

        file_bytes = await file.read()

        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            logger.warning(
                f"Rejected upload: size={len(file_bytes)} bytes "
                f"exceeds {MAX_ANALYSIS_SIZE_BYTES} bytes"
            )
            raise HTTPException(
                status_code=413,
                detail=f"Payload too large. "
                       f"Max size: {MAX_ANALYSIS_SIZE_BYTES // (1024*1024)}MB"
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

        # Quality gate — reject images too small or unreadable for forensics
        from backend.utils.image_quality import assess_image_quality
        quality = assess_image_quality(file_bytes, file.filename)
        if not quality["suitable"]:
            raise HTTPException(
                status_code=422,
                detail=f"Image unsuitable for analysis: {quality['reason']}"
            )

        cached_result = forensics_cache.get(file_hash)
        if cached_result:
            logger.info(
                f"Cache HIT: Returning cached result for {file.filename} "
                f"(saved ~2-5 seconds of processing)"
            )
            return cached_result

        logger.info(f"Cache MISS: Running full analysis for {file.filename}")
        forensics = ImageForensics(file_bytes, file.filename)
        report = forensics.generate_forensic_report()

        forensics_cache.set(file_hash, report)

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
        def _sanitize(obj):
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

        return report

    except FileValidationError as e:
        logger.warning("Validation failed: %s", str(e))
        raise HTTPException(status_code=422, detail="File validation failed. Check file type and size.")

    except ValueError as e:
        logger.error("Value error during analysis: %s", str(e))
        raise HTTPException(status_code=400, detail="Invalid image data")

    except IOError as e:
        logger.error("I/O error during analysis: %s", str(e))
        raise HTTPException(status_code=422, detail="Unable to process image")

    except HTTPException:
        raise

    except Exception:
        logger.error("Unexpected analysis error: %s", exc_info=True)
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
            raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")

        file_bytes = await file.read()

        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")

        result = generate_heatmap(file_bytes, file.filename)

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
        logger.error("Heatmap generation error: %s", exc_info=True)
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
            raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")

        file_bytes = await file.read()

        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")

        result = attribute_generator(file_bytes, file.filename)

        logger.info(
            f"Attribution complete: file={file.filename}, "
            f"generator={result['predicted_generator']}, "
            f"confidence={result['confidence']:.3f}"
        )

        return result

    except HTTPException:
        raise
    except Exception:
        logger.error("Attribution error: %s", exc_info=True)
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
            raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")

        file_bytes = await file.read()

        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")

        result = detect_platform(file_bytes, file.filename)

        logger.info(
            f"Platform detection: file={file.filename}, "
            f"platform={result['predicted_platform']}, "
            f"confidence={result['confidence']:.3f}"
        )

        return result

    except HTTPException:
        raise
    except Exception:
        logger.error("Platform detection error: %s", exc_info=True)
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
            raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")
        file_bytes = await file.read()
        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")
        result = verify_c2pa(file_bytes, file.filename)
        logger.info(
            f"C2PA verification: file={file.filename}, "
            f"status={result['provenance_status']}, "
            f"has_c2pa={result['has_c2pa']}"
        )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.error("C2PA verification error: %s", exc_info=True)
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
            raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")
        file_bytes = await file.read()
        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")
        result = run_robustness_test(file_bytes, file.filename)
        logger.info(
            f"Robustness test: file={file.filename}, "
            f"overall={result.get('overall_robustness', 0):.3f}, "
            f"level={result.get('robustness_level', 'unknown')}"
        )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.error("Robustness test error: %s", exc_info=True)
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
        result = process_batch(images)

        logger.info(
            f"Batch complete: processed={result.get('processed', 0)}, "
            f"failed={result.get('failed', 0)}, "
            f"verdict={result.get('statistics', {}).get('batch_verdict', 'unknown')}"
        )

        return result

    except HTTPException:
        raise
    except Exception:
        logger.error("Batch analysis error: %s", exc_info=True)
        raise HTTPException(status_code=500, detail="Batch analysis failed")
    finally:
        for file in files:
            await file.close()


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
            raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")
        file_bytes = await file.read()
        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large. Max 10MB.")

        from backend.services.image_forensics import ImageForensics
        forensics = ImageForensics(file_bytes, file.filename)
        report    = forensics.generate_forensic_report()

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
        logger.error("Export error: %s", exc_info=True)
        raise HTTPException(status_code=500, detail="Export generation failed. Please try again.")
    finally:
        await file.close()
