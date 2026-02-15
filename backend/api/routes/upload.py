"""
File upload endpoints.
Why: Separate routing from main.py for scalability.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.utils.validators import validate_file, FileValidationError
from backend.models.schemas import FileValidationResponse, ErrorResponse
from backend.core.logger import setup_logger

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
    - Images: JPEG, PNG, WebP
    - Videos: MP4, MPEG
    - Documents: PDF
    
    **Max size:** 50MB
    """
)
async def validate_upload(
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
        
    except Exception as e:
        logger.error(f"Unexpected error validating {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during file validation"
        )
    
    finally:
        # Ensure file is closed (privacy - no disk storage)
        await file.close()
