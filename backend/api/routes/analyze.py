"""
Forensic analysis endpoints.
Security: Rate limited, validated, privacy-first.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
import hashlib

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

        return report

    except FileValidationError as e:
        logger.warning(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

    except ValueError as e:
        logger.error(f"Value error during analysis: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image data")

    except IOError as e:
        logger.error(f"I/O error during analysis: {str(e)}")
        raise HTTPException(status_code=422, detail="Unable to process image")

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected analysis error: {str(e)}", exc_info=True)
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
    except Exception as e:
        logger.error(f"Heatmap generation error: {e}", exc_info=True)
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
    except Exception as e:
        logger.error(f"Attribution error: {e}", exc_info=True)
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
    except Exception as e:
        logger.error(f"Platform detection error: {e}", exc_info=True)
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
    except Exception as e:
        logger.error(f"C2PA verification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="C2PA verification failed")
    finally:
        await file.close()
