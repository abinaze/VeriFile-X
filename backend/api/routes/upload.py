"""
File upload endpoints.
Why: Separate routing from main.py for scalability.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from backend.utils.validators import validate_file, FileValidationError
from backend.models.schemas import FileValidationResponse, ErrorResponse
from backend.core.logger import setup_logger
from backend.core.config import settings

logger = setup_logger(__name__)

# Create router (will be included in main.py)
router = APIRouter(
    prefix="/api/v1/upload",
    tags=["File Upload"]
)


@router.post(
    "/validate",
    response_model=FileValidationResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Validate uploaded file",
    description="""
    Validates file type and size without processing.
    
    **Privacy:** File is read into memory and immediately discarded.
    No files are stored on disk.
    
    **Supported types:**
    - Images: JPEG, PNG, WebP, TIFF, HEIC/HEIF

    Note: video and document upload support is not currently enabled
    (ALLOWED_VIDEO_TYPES/ALLOWED_DOC_TYPES are empty in this deployment's
    settings) — image-only for now.

    **Max size:** see MAX_FILE_SIZE_MB setting.
    """
)
async def validate_upload(
    request: Request,
    file: UploadFile = File(..., description="File to validate")
):
    """
    Validate uploaded file.
    
    Why async?
    - Non-blocking file read
    - Handles multiple concurrent uploads
    - FastAPI best practice
    """
    try:
        # Check Content-Length header BEFORE reading (F-15) -- this
        # endpoint is intentionally public (no API key required) and had
        # no size check at all before the full body was read into memory,
        # unlike /api/v1/analyze/image which has this same check.
        # Client-supplied and absent under chunked transfer, so this is a
        # real improvement, not a complete guarantee -- validate_file()'s
        # own post-read size check (below, via file_bytes length) remains
        # the authoritative one.
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Payload too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
                    )
            except ValueError:
                pass  # Invalid Content-Length header — proceed and check after read

        # Read file into memory
        file_bytes = await file.read()
        logger.info(f"Received file: {file.filename}, size: {len(file_bytes)} bytes")
        
        # Validate
        result = validate_file(file_bytes, file.filename)
        
        logger.info(f"Validation successful: {result}")
        return FileValidationResponse(**result)
        
    except FileValidationError as e:
        logger.warning(f"Validation failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        # Let our own 413 (Content-Length too large) and any other
        # intentional HTTPException propagate unchanged instead of being
        # caught and rewritten as a generic 500 by the except-Exception
        # clause below.
        raise

    except Exception as e:
        logger.error(f"Unexpected error validating {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during file validation"
        )
    
    finally:
        # Ensure file is closed (privacy - no disk storage)
        await file.close()
