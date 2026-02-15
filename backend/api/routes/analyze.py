"""
Forensic analysis endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.services.image_forensics import ImageForensics
from backend.models.forensics import ForensicReport
from backend.utils.validators import validate_file, FileValidationError
from backend.core.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(
    prefix="/api/v1/analyze",
    tags=["Forensic Analysis"]
)


@router.post(
    "/image",
    response_model=ForensicReport,
    summary="Analyze image forensics",
    description="""
    Performs comprehensive forensic analysis on uploaded image.
    
    **Analysis includes:**
    - EXIF metadata extraction (camera, GPS, timestamps)
    - Cryptographic hash generation (SHA-256, MD5)
    - Perceptual hashing for similarity detection
    - Tampering indicator detection
    - Authenticity confidence scoring
    
    **Privacy:** File processed in-memory, immediately discarded.
    
    **Supported formats:** JPEG, PNG, WebP
    """
)
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze")
):
    """
    Perform forensic analysis on uploaded image.
    """
    try:
        # Read and validate file
        file_bytes = await file.read()
        logger.info(f"Analyzing image: {file.filename}")
        
        # Validate file type and size
        validation = validate_file(file_bytes, file.filename)
        
        # Check if it's actually an image
        if not validation["mime_type"].startswith("image/"):
            raise FileValidationError("File must be an image (JPEG, PNG, or WebP)")
        
        # Perform forensic analysis
        forensics = ImageForensics(file_bytes, file.filename)
        report = forensics.generate_forensic_report()
        
        logger.info(f"Analysis complete for {file.filename}")
        return report
        
    except FileValidationError as e:
        logger.warning(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
    
    finally:
        await file.close()
