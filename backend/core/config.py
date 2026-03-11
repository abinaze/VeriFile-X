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
    
    ALLOWED_IMAGE_TYPES: set[str] = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }
    
    ALLOWED_VIDEO_TYPES: set[str] = set()  # Empty set - no video support yet
    
    ALLOWED_UPLOAD_EXTENSIONS: set[str] = {
        ".jpg", ".jpeg", ".png", ".webp"
    }
    
    # =========================================================================
    # CORS CONFIGURATION
    # =========================================================================
    
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    # Debug Mode
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "VeriFile-X"
    VERSION: str = "6.0.0"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    
    # File Upload Limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    MAX_ANALYSIS_SIZE_MB: int = int(os.getenv("MAX_ANALYSIS_SIZE_MB", "10"))
    
    # Cache Settings
    CACHE_TTL_MINUTES: int = int(os.getenv("CACHE_TTL_MINUTES", "60"))
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", "500"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
