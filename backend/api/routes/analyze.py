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
from backend.utils.image_quality import assess_image_quality
from backend.core.logger import setup_logger
from backend.core.cache import forensics_cache
from backend.services.metrics_collector import record_analysis
from backend.core.audit_log import log_analysis
from backend.core.config import settings
from backend.utils.json_safe import sanitize as _sanitize

logger = setup_logger(__name__)


def _correct_exif_orientation(file_bytes: bytes) -> bytes:
    """Apply EXIF orientation before all forensic analysis.

    iPhone portrait photos have EXIF rotation=6 (90 deg CW) -- without
    this, all spatial signals (PRNU, CFA, noise map) run on a sideways
    image.

    C-2 fix: the previous version unconditionally re-encoded EVERY
    upload through this step, even when no rotation was needed --
    ImageOps.exif_transpose() returns a fresh copy even when the
    orientation tag is 1/absent (confirmed by direct reproduction) --
    and the re-encode passed no quality= argument, silently defaulting
    to Pillow's JPEG quality=75 regardless of the original quality.
    Measured impact on a real quality-95 synthetic photo: ~55% smaller,
    mean pixel error 12.65/255 -- a large, easily-detectable
    perturbation to the ~31% of ensemble weight (ELA, JPEG Ghost, DCT,
    PRNU-heuristic, Noise Map, Noiseprint, CFA) that measures exactly
    this kind of compression artifact.

    Fix: (1) only re-encode when the Orientation tag is actually present
    and not the identity value (1) -- skip re-encoding entirely
    otherwise, so the vast majority of uploads are analyzed completely
    untouched; (2) when re-encoding IS needed, capture the original
    JPEG's quantization tables and chroma subsampling BEFORE calling
    exif_transpose() and pass them through explicitly.
    ImageOps.exif_transpose() returns a plain PIL.Image.Image (not a
    JpegImageFile), so it has no .quantization attribute left for
    Pillow's quality="keep"/qtables="keep" shortcut to use once called
    (confirmed by direct testing) -- the tables must be captured from
    the ORIGINAL image object first.

    Extracted to a standalone, directly-unit-testable function (was
    previously inline in analyze_image()) specifically so this fix has
    real regression coverage independent of the full detection pipeline.

    Returns the original bytes unchanged, or on any error, per the
    pre-existing fail-open behavior (proceed with original bytes if EXIF
    correction fails for any reason).
    """
    try:
        from PIL import Image as _PIL_img, ImageOps as _EXIF_ops
        from io import BytesIO as _BytesIO_exif

        _img_exif = _PIL_img.open(_BytesIO_exif(file_bytes))
        _orig_fmt = _img_exif.format or "JPEG"
        _orientation = _img_exif.getexif().get(0x0112, 1)  # 0x0112 == Orientation

        if _orientation in (1, None):
            # No rotation needed -- return the upload byte-for-byte
            # untouched, so compression-sensitive signals measure the
            # actual upload, not an incidental recompression.
            return file_bytes

        _orig_qtables = None
        _orig_subsampling = None
        if _orig_fmt == "JPEG":
            _orig_qtables = getattr(_img_exif, "quantization", None)
            try:
                from PIL.JpegImagePlugin import get_sampling as _get_sampling
                _orig_subsampling = _get_sampling(_img_exif)
            except Exception:
                _orig_subsampling = None

        _img_exif = _EXIF_ops.exif_transpose(_img_exif)
        _buf_exif = _BytesIO_exif()
        _save_kwargs = {"format": _orig_fmt}
        if _orig_fmt == "JPEG" and _orig_qtables:
            _save_kwargs["qtables"] = list(_orig_qtables.values())
            if _orig_subsampling is not None:
                _save_kwargs["subsampling"] = _orig_subsampling
        _img_exif.save(_buf_exif, **_save_kwargs)
        return _buf_exif.getvalue()
    except Exception:
        return file_bytes  # Proceed with original bytes if EXIF correction fails


# Standalone limiter for this router — same config as main.py
# Wires RATE_LIMIT_PER_MINUTE as the default limit for any endpoint
# without its own explicit @limiter.limit(...) decorator.
from backend.core.config import settings as _settings
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{_settings.RATE_LIMIT_PER_MINUTE}/minute"])

from backend.core.auth import require_role_for_method_or_demo
from fastapi import Depends

router = APIRouter(
    prefix="/api/v1/analyze",
    # F-5 extension: was require_analyst_or_demo (any analyst/admin key,
    # any method) -- now the same per-method ROLES enforcement cases.py
    # has, still with the F-1 public demo bypass. A viewer-role key can
    # now reach GET-only endpoints here (/history, /stats) but still
    # correctly gets 403 on the POST analysis endpoints.
    dependencies=[Depends(require_role_for_method_or_demo)],
    tags=["Forensic Analysis"]
)

# Allowed MIME types for analysis endpoint
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/heic", "image/heif"}

# Max file size for analysis (CPU-intensive operation). Previously hardcoded
# to 10MB independently of settings.MAX_ANALYSIS_SIZE_MB, so changing the
# config value had no effect on the actual enforced limit. Now derived from
# settings so there is a single source of truth.
MAX_ANALYSIS_SIZE_BYTES = settings.MAX_ANALYSIS_SIZE_MB * 1024 * 1024


def _reject_if_content_length_exceeds(request: Request, max_bytes: int) -> None:
    """C-3 (resource-exhaustion stopgap): reject a request BEFORE its body
    is read into memory, using the Content-Length header, whenever the
    client declares a size over the limit.

    Seven of this router's nine file-accepting endpoints previously read
    the entire request body via `await file.read()` before checking its
    size at all -- meaning a very large upload was fully buffered into
    memory before ever being rejected. Only POST /image had this
    pre-check; this helper lets every sibling endpoint apply the same
    guard consistently without duplicating the same few lines nine times.

    This is a stopgap, not a full fix: Content-Length is client-supplied
    and can be absent, wrong, or (for chunked transfer encoding) simply
    not sent -- in all of those cases this check silently no-ops and the
    existing post-read size check (unchanged, still present on every
    endpoint) remains the actual backstop. A real fix would enforce a
    hard cap while STREAMING the body, which is a larger change tracked
    separately as part of consolidating upload validation into one
    shared dependency used by all nine endpoints.
    """
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Payload too large. Max size: {max_bytes // (1024*1024)}MB"
                )
        except ValueError:
            pass  # Invalid Content-Length header -- proceed, the post-read check still applies


def _prepare_image_bytes(
    file_bytes: bytes,
    filename: str,
    max_bytes: int,
    correct_exif: bool = True,
):
    """C-3 (full fix): the shared per-file-content half of the upload
    pipeline -- optional EXIF-orientation correction, a post-correction
    size re-check, MIME/extension validation, and the image-quality gate.

    Before this, POST /image was the only one of this router's 9
    file-accepting endpoints with all of these checks; the other 8 had an
    inconsistent subset (see this file's module docstring history / the
    audit's C-3 finding). This function is the single implementation the
    other 8 now also call, via prepare_upload() below (or directly, for
    /batch -- see analyze_batch()'s own comment for why it's the one
    exception).

    correct_exif=False exists specifically for /platform and /c2pa: both
    fingerprint properties of the ORIGINAL file structure -- EXIF
    presence/absence, and a binary JUMBF C2PA manifest respectively --
    that even a careful, quality-preserving EXIF re-encode would corrupt
    or destroy outright. Removing this parameter to "simplify" the
    pipeline would silently reintroduce that bug. See analyze_platform()
    and analyze_c2pa() below for the full rationale.

    Returns (prepared_bytes, confidence_cap). Raises HTTPException(413)
    if the (possibly re-encoded) bytes exceed max_bytes, or
    HTTPException(422) if the file fails MIME/extension validation or the
    quality gate. A missing/empty filename defaults to "upload" (matching
    the fallback several endpoints already used ad hoc), rather than
    letting validate_file()'s extension parsing fail on None.
    """
    filename = filename or "upload"

    if correct_exif:
        file_bytes = _correct_exif_orientation(file_bytes)

    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Payload too large. Max size: {max_bytes // (1024*1024)}MB"
        )

    try:
        validation = validate_file(file_bytes, filename)
    except FileValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if not validation["mime_type"].startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail=f"File content is not an image: {validation['mime_type']}"
        )
    if not validation.get("extension_valid", True):
        raise HTTPException(
            status_code=422,
            detail=f"File extension not allowed: {validation.get('extension', 'unknown')}"
        )

    quality = assess_image_quality(file_bytes, filename)
    if not quality["suitable"]:
        raise HTTPException(
            status_code=422,
            detail=f"Image unsuitable for analysis: {quality['reason']}"
        )

    return file_bytes, quality.get("confidence_cap", 1.0)


async def prepare_upload(
    request: Request,
    file: UploadFile,
    max_bytes: int,
    correct_exif: bool = True,
):
    """C-3 (full fix): the full shared upload pipeline -- content-type
    header check (415) -> pre-read Content-Length guard (413) -> read
    the body -> _prepare_image_bytes() (EXIF correction, post-read size
    check, MIME/extension validation, quality gate).

    Used by every file-accepting endpoint in this router except /batch,
    which calls _prepare_image_bytes() directly per file inside its own
    loop instead -- a per-file call here would re-check this same
    whole-request Content-Length header against a much smaller per-file
    threshold on every iteration of that loop, incorrectly rejecting
    almost any real multi-file batch. See analyze_batch() below.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type. Allowed: image/jpeg, image/png, image/webp"
        )

    _reject_if_content_length_exceeds(request, max_bytes)
    file_bytes = await file.read()

    return _prepare_image_bytes(file_bytes, file.filename, max_bytes, correct_exif=correct_exif)


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
        # C-3 (full fix): all of content-type check, pre-read Content-Length
        # guard, EXIF-orientation correction, post-read size check,
        # MIME/extension validation, and the quality gate now live in the
        # shared prepare_upload() pipeline -- see its docstring above.
        # This endpoint was the only one of the 9 with all six checks;
        # the other 8 are migrated onto the same shared calls below it in
        # this file.
        file_bytes, _confidence_cap = await prepare_upload(
            request, file, MAX_ANALYSIS_SIZE_BYTES, correct_exif=True
        )

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        logger.info(
            f"Analyzing: file={file.filename}, "
            f"size={len(file_bytes)} bytes, "
            f"sha256={file_hash[:16]}..., "
            f"content_type={file.content_type}"
        )

        cached_result = forensics_cache.get(file_hash)
        if cached_result:
            logger.info(
                f"Cache HIT: Returning cached result for {file.filename} "
                f"(saved ~2-5 seconds of processing)"
            )
            # No re-sanitization needed here (F-8/F-27): every report that
            # enters forensics_cache has already been through _sanitize()
            # before caching, below.
            return cached_result

        logger.info(f"Cache MISS: Running full analysis for {file.filename}")
        import asyncio as _asyncio
        _t0 = time.perf_counter()
        forensics = ImageForensics(file_bytes, file.filename)
        report = await _asyncio.get_running_loop().run_in_executor(
            None, forensics.generate_forensic_report
        )
        _latency_ms = round((time.perf_counter() - _t0) * 1000, 1)

        # Sanitize any NaN/Inf float values BEFORE this report is cached,
        # sent to any webhook, or written to the audit log (F-8) -- moved
        # here from just before the return statement, where it protected
        # only the direct HTTP response. forensics_cache.set(), fire_webhooks(),
        # and log_analysis() below used to all see the raw, unsanitized report.
        report = _sanitize(report)

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

        # (NaN/Infinity sanitization now happens immediately after report
        # generation, above -- see F-8/F-27 -- so no second pass is needed
        # here before returning.)

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
        # C-3 (full fix): was content-type + size checks only -- now also
        # gets EXIF correction, MIME/extension validation, and the quality
        # gate via the shared prepare_upload() pipeline.
        file_bytes, _ = await prepare_upload(
            request, file, MAX_ANALYSIS_SIZE_BYTES, correct_exif=True
        )

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
        # C-3 (full fix): was content-type + size checks only -- now also
        # gets EXIF correction, MIME/extension validation, and the quality
        # gate via the shared prepare_upload() pipeline.
        file_bytes, _ = await prepare_upload(
            request, file, MAX_ANALYSIS_SIZE_BYTES, correct_exif=True
        )

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
        # C-3 (full fix): correct_exif=False is deliberate, not an
        # oversight -- this endpoint keys partly on EXIF presence/absence
        # (WhatsApp/Instagram/Telegram are known to strip EXIF; other
        # platforms don't), and exif_transpose() strips the orientation
        # tag as part of "correcting" it -- which would alter exactly the
        # signal this endpoint exists to measure, for exactly the photos
        # most likely to need rotation. Still gains MIME/extension
        # validation and the quality gate via prepare_upload().
        file_bytes, _ = await prepare_upload(
            request, file, MAX_ANALYSIS_SIZE_BYTES, correct_exif=False
        )

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
        # C-3 (full fix): correct_exif=False is deliberate -- this
        # endpoint looks for a binary JUMBF box containing a
        # cryptographically-signed provenance manifest. A generic Pillow
        # re-encode has no concept of this custom segment and will not
        # preserve it -- "correcting" a real, C2PA-signed rotated photo
        # would silently destroy the ability to verify it, the exact
        # opposite of what a provenance-verification endpoint should do
        # to its own input. Still gains MIME/extension validation and the
        # quality gate via prepare_upload().
        file_bytes, _ = await prepare_upload(
            request, file, MAX_ANALYSIS_SIZE_BYTES, correct_exif=False
        )
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
        # C-3 (full fix): was content-type + size checks only -- now also
        # gets EXIF correction, MIME/extension validation, and the quality
        # gate via the shared prepare_upload() pipeline.
        file_bytes, _ = await prepare_upload(
            request, file, MAX_ANALYSIS_SIZE_BYTES, correct_exif=True
        )
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
    from backend.services.batch_processor import process_batch, MAX_BATCH_SIZE, MAX_IMAGE_BYTES

    try:
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Too many files. Max {MAX_BATCH_SIZE} per batch request."
            )

        # C-3 fix: this endpoint previously had NO size check anywhere in the
        # route handler at all -- the 5MB MAX_IMAGE_BYTES cap only existed
        # inside batch_processor.process_batch(), checked only after all
        # (up to 10) files were already fully buffered into memory. A coarse
        # pre-read guard on the whole request's declared size catches the
        # obviously-oversized case before any file is read; a genuine
        # per-file check (below) rejects individually-oversized files
        # before they're retained for the downstream pipeline.
        _reject_if_content_length_exceeds(request, MAX_BATCH_SIZE * MAX_IMAGE_BYTES)

        images = []
        for file in files:
            if file.content_type not in ALLOWED_IMAGE_TYPES:
                logger.warning(f"Batch: skipping {file.filename} — unsupported type {file.content_type}")
                continue
            data = await file.read()
            if len(data) > MAX_IMAGE_BYTES:
                logger.warning(
                    f"Batch: skipping {file.filename} — "
                    f"{len(data)} bytes exceeds {MAX_IMAGE_BYTES} byte per-image limit"
                )
                continue
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
        # C-3 fix: this endpoint previously relied solely on validate_file()'s
        # general-purpose 50MB (MAX_FILE_SIZE_MB) ceiling -- five times higher
        # than the 10MB (MAX_ANALYSIS_SIZE_MB) limit every sibling CPU-heavy
        # analysis endpoint enforces. Both the pre-read Content-Length guard
        # and an explicit post-read check now use the same, intended ceiling.
        _reject_if_content_length_exceeds(request, MAX_ANALYSIS_SIZE_BYTES)
        image_bytes = await file.read()
        if len(image_bytes) > MAX_ANALYSIS_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Payload too large. Max size: {MAX_ANALYSIS_SIZE_BYTES // (1024*1024)}MB"
            )
        from backend.utils.validators import validate_file, FileValidationError as _FVE
        validate_file(image_bytes, file.filename or "upload")
    except HTTPException:
        raise
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
    finally:
        # C-3 fix: this was the only file-accepting endpoint in this router
        # that never closed its UploadFile in a finally block.
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
            raise HTTPException(status_code=415, detail="Unsupported media type. Allowed: image/jpeg, image/png, image/webp")
        _reject_if_content_length_exceeds(request, MAX_ANALYSIS_SIZE_BYTES)
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
            # Same sanitize-before-cache fix as F-8 above: this endpoint has
            # its own separate cache lookup/report-generation path and was
            # not sanitizing at all -- export_json()'s plain json.dumps()
            # would emit the literal (invalid-per-RFC-8259) NaN/Infinity
            # tokens for any non-finite signal score.
            report = _sanitize(report)
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

    _reject_if_content_length_exceeds(request, MAX_ANALYSIS_SIZE_BYTES)
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
