"""
Application configuration with environment variable support.
"""
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """
    Application settings with environment variable overrides.
    
    Environment variables take precedence over defaults.
    For production, create a .env file based on .env.example
    """
    
    # =========================================================================
    # FILE TYPE VALIDATION
    # =========================================================================
    
    # Image types (supported for forensic analysis)
    ALLOWED_IMAGE_TYPES: set[str] = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }
    
    # Video types (not currently supported - empty set)
    ALLOWED_VIDEO_TYPES: set[str] = set()
    
    # Document types (not currently supported - empty set)
    ALLOWED_DOC_TYPES: set[str] = set()
    
    # Upload file extensions (must match MIME types above)
    ALLOWED_UPLOAD_EXTENSIONS: set[str] = {
        ".jpg", ".jpeg", ".png", ".webp"
    }
    
    # =========================================================================
    # FILE SIZE LIMITS
    # =========================================================================
    
    # Maximum file size for upload (in MB)
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    # Maximum file size for analysis (in MB) - smaller than upload
    MAX_ANALYSIS_SIZE_MB: int = int(os.getenv("MAX_ANALYSIS_SIZE_MB", "10"))

    # =========================================================================
    # CORS CONFIGURATION
    # =========================================================================

    # CORS allowed origins (comma-separated in env var)
    # Development default: localhost only
    # Production: Set via CORS_ORIGINS environment variable
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000")

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


    # DEBUG MODE
    # IMPORTANT: Set DEBUG=False in production!
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"


    # API CONFIGURATION
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "VeriFile-X"
    VERSION: str = "6.0.0"


    # RATE LIMITING
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))


    # CACHE SETTINGS
    CACHE_TTL_MINUTES: int = int(os.getenv("CACHE_TTL_MINUTES", "60"))
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", "500"))


    # LOGGING
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        case_sensitive = True



settings = Settings()
