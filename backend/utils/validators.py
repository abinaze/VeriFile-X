"""
File validation utilities for secure file processing.
Why: Prevent malicious files, enforce size limits, validate MIME types.
"""
import magic
from typing import Tuple
from backend.core.config import settings
from backend.core.logger import setup_logger

logger = setup_logger(__name__)

# Extensions that are valid for the allowed MIME types
_ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "webp", "tiff", "tif", "heic", "heif",
    "mp4", "mov", "avi", "mkv",
    "pdf", "doc", "docx",
}


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
    Complete file validation (type + size + extension).
    """
    # Validate size first (fail fast for DoS protection)
    size_bytes = validate_file_size(file_bytes, filename)

    # Then validate type
    mime_type, extension = validate_file_type(file_bytes, filename)

    # Extension cross-check — catches files whose content MIME passes but
    # extension was crafted to bypass downstream extension-based filters.
    extension_valid = (extension in _ALLOWED_EXTENSIONS) if extension else True

    return {
        "valid":           True,
        "mime_type":       mime_type,
        "extension":       extension,
        "extension_valid": extension_valid,
        "size_bytes":      size_bytes,
        "size_mb":         round(size_bytes / (1024 * 1024), 2),
        "filename":        filename,
    }
