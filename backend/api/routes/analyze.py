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
        # Layer 1: Content-type header check (fast fail)
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            logger.warning(
                f"Rejected upload: content_type={file.content_type} "
                f"from {get_remote_address(request)}"
            )
            raise HTTPException(
                status_code=415,  # FIXED: Changed from 400 to 415 (Unsupported Media Type)
                detail=f"Unsupported media type: {file.content_type}. "
                       f"Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}"
            )

        # Read file into memory
        file_bytes = await file.read()

        # Layer 2: Actual size check (after read)
        if len(file_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            logger.warning(
                f"Rejected upload: size={len(file_bytes)} bytes "
                f"exceeds {MAX_ANALYSIS_SIZE_BYTES} bytes"
            )
            raise HTTPException(
                status_code=413,  # FIXED: Changed from 400 to 413 (Payload Too Large)
                detail=f"Payload too large. "
                       f"Max size: {MAX_ANALYSIS_SIZE_BYTES // (1024*1024)}MB"
            )

        # OPTIMIZATION: Compute SHA-256 once for caching
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        logger.info(
            f"Analyzing: file={file.filename}, "
            f"size={len(file_bytes)} bytes, "
            f"sha256={file_hash[:16]}..., "
            f"content_type={file.content_type}"
        )

        # Layer 3: MIME type validation via python-magic
        validation = validate_file(file_bytes, file.filename)

        if not validation["mime_type"].startswith("image/"):
            raise FileValidationError(
                f"File content is not an image: {validation['mime_type']}"
            )

        # Layer 4: Check cache (skip expensive analysis if duplicate)
        cached_result = forensics_cache.get(file_hash)
        if cached_result:
            logger.info(
                f"Cache HIT: Returning cached result for {file.filename} "
                f"(saved ~2-5 seconds of processing)"
            )
            return cached_result

        # Cache miss - run full forensic analysis
        logger.info(f"Cache MISS: Running full analysis for {file.filename}")
        forensics = ImageForensics(file_bytes, file.filename)
        report = forensics.generate_forensic_report()

        # Store in cache for future duplicate uploads
        forensics_cache.set(file_hash, report)

        logger.info(
            f"Analysis complete: file={file.filename}, "
            f"ai_probability={report['summary']['ai_probability']:.3f}, "
            f"classification={report['summary']['ai_classification']}"
        )

        return report

    except FileValidationError as e:
        logger.warning(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))  # IMPROVED: 422 for validation

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
