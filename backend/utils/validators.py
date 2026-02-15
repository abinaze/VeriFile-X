"""
File validation utilities for secure file processing.
Why: Prevent malicious files, enforce size limits, validate MIME types.
"""
import magic
from typing import Tuple, Optional
from backend.core.config import settings
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


class FileValidationError(Exception):
    """Custom exception for file validation failures."""
    pass


def validate_file_type(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Validate file MIME type using python-magic (reads file signature).
    
    Why python-magic? 
    - Reads actual file headers, not just extension
    - Prevents .exe renamed to .jpg attacks
    
    Args:
        file_bytes: Raw file content
        filename: Original filename (for extension check)
        
    Returns:
        Tuple of (mime_type, file_extension)
        
    Raises:
        FileValidationError: If file type not allowed
    """
    # Get MIME type from file content (not extension)
    mime = magic.Magic(mime=True)
    detected_mime = mime.from_buffer(file_bytes)
    
    logger.info(f"File '{filename}' detected as {detected_mime}")
    
    # Combine all allowed types
    allowed_types = (
        settings.ALLOWED_IMAGE_TYPES +
        settings.ALLOWED_VIDEO_TYPES +
        settings.ALLOWED_DOC_TYPES
    )
    
    if detected_mime not in allowed_types:
        raise FileValidationError(
            f"File type '{detected_mime}' not allowed. "
            f"Allowed: {', '.join(allowed_types)}"
        )
    
    # Extract extension
    extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    
    return detected_mime, extension


def validate_file_size(file_bytes: bytes, filename: str) -> int:
    """
    Validate file size against limit.
    
    Why size limits?
    - Prevent DoS attacks (100GB upload)
    - Memory constraints (in-memory processing)
    - Reasonable for forensic analysis
    
    Args:
        file_bytes: Raw file content
        filename: Original filename (for logging)
        
    Returns:
        File size in bytes
        
    Raises:
        FileValidationError: If file exceeds limit
    """
    size_bytes = len(file_bytes)
    size_mb = size_bytes / (1024 * 1024)
    max_size_mb = settings.MAX_FILE_SIZE_MB
    
    logger.info(f"File '{filename}' size: {size_mb:.2f} MB")
    
    if size_mb > max_size_mb:
        raise FileValidationError(
            f"File size ({size_mb:.2f} MB) exceeds limit ({max_size_mb} MB)"
        )
    
    return size_bytes


def validate_file(file_bytes: bytes, filename: str) -> dict:
    """
    Complete file validation (type + size).
    """

    # Validate size first (fail fast for DoS protection)
    size_bytes = validate_file_size(file_bytes, filename)

    # Then validate type
    mime_type, extension = validate_file_type(file_bytes, filename)

    return {
        "valid": True,
        "mime_type": mime_type,
        "extension": extension,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "filename": filename
    }
