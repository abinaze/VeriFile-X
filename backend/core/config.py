"""
Application configuration management.
Uses environment variables for security-sensitive settings.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Why Pydantic? Type validation, auto-documentation, easy testing.
    """
    # API Settings
    API_TITLE: str = "VeriFile-X API"
    API_VERSION: str = "0.1.0"
    API_DESCRIPTION: str = "Privacy-preserving digital forensics platform"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # File Processing Limits (privacy + performance)
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/webp"]
    ALLOWED_VIDEO_TYPES: list = ["video/mp4", "video/mpeg"]
    ALLOWED_DOC_TYPES: list = ["application/pdf"]
    
    # Security
    CORS_ORIGINS: list = ["http://localhost:3000"]  # Frontend URLs
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton pattern - one instance across app
settings = Settings()
