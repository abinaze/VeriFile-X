"""
Tests for file validation utilities.
"""
import pytest
from backend.utils.validators import (
    validate_file_type,
    validate_file_size,
    validate_file,
    FileValidationError
)


def test_validate_file_type_valid_png(sample_image_bytes):
    mime_type, extension = validate_file_type(sample_image_bytes, "test.png")
    assert mime_type == "image/png"
    assert extension == "png"


def test_validate_file_type_invalid():
    fake_image = b"This is not an image"

    with pytest.raises(FileValidationError):
        validate_file_type(fake_image, "fake.jpg")


def test_validate_file_size_within_limit(sample_image_bytes):
    size = validate_file_size(sample_image_bytes, "small.png")
    assert size == len(sample_image_bytes)


def test_validate_file_size_exceeds_limit():
    large_file = b"x" * (60 * 1024 * 1024)

    with pytest.raises(FileValidationError):
        validate_file_size(large_file, "huge.bin")


def test_validate_file_complete(sample_image_bytes):
    result = validate_file(sample_image_bytes, "test.png")

    assert result["valid"] is True
    assert result["mime_type"] == "image/png"
    assert result["extension"] == "png"
    assert result["size_bytes"] > 0
    assert result["size_mb"] < 0.01
    assert result["filename"] == "test.png"

