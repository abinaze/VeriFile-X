"""
Application configuration with environment variable support.

All settings are loaded by pydantic-settings from environment variables
or a .env file. Do NOT use os.getenv() as default values inside this class
— that bypasses pydantic-settings loading and prevents .env overrides.
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """
    Application settings loaded via pydantic-settings.

    Override any value by setting the corresponding environment variable
    or adding it to a .env file in the project root.
    """

    # =========================================================================
    # FILE TYPE VALIDATION
    # =========================================================================

    ALLOWED_IMAGE_TYPES: tuple = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/tiff",
        "image/heic",
        "image/heif",
    )

    ALLOWED_VIDEO_TYPES: tuple = ()
    ALLOWED_DOC_TYPES: tuple   = ()

    ALLOWED_UPLOAD_EXTENSIONS: tuple = (
        ".jpg", ".jpeg", ".png", ".webp",
        ".tiff", ".tif", ".heic", ".heif"
    )

    # =========================================================================
    # CORS CONFIGURATION
    # =========================================================================

    CORS_ORIGINS: str = (
        "http://localhost:3000,"
        "http://localhost:8000,"
        "http://localhost:8080,"
        "http://127.0.0.1:8000,"
        "https://abinaze.github.io,"
        "https://abinazebinoy-verifile-x-api.hf.space"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ORIGINS into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    # =========================================================================
    # APPLICATION
    # =========================================================================

    DEBUG:        bool = False
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME:  str = "VeriFile-X"
    VERSION:       str = "8.4.0"

    # =========================================================================
    # RATE LIMITING
    # =========================================================================

    RATE_LIMIT_PER_MINUTE: int = 10

    # =========================================================================
    # FILE UPLOAD LIMITS
    # =========================================================================

    MAX_FILE_SIZE_MB:     int = 50
    MAX_ANALYSIS_SIZE_MB: int = 10

    # =========================================================================
    # CACHE SETTINGS
    # =========================================================================

    CACHE_TTL_MINUTES: int = 60
    MAX_CACHE_SIZE:    int = 500

    # =========================================================================
    # LOGGING
    # =========================================================================

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file       = ".env"
        case_sensitive = True


settings = Settings()
