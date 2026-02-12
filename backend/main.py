"""
VeriFile-X API - Privacy-preserving digital forensics platform.

Architecture:
- No file storage (all processing in-memory)
- Modular services for different analysis types
- Type-safe request/response models
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from backend.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    Why: Resource initialization (models, connections) and cleanup.
    """
    logger.info("🚀 VeriFile-X API starting up...")
    # TODO Phase 5: Load AI models here
    yield
    logger.info("🛑 VeriFile-X API shutting down...")
    # TODO: Cleanup resources


# Initialize FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
)

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "operational",
        "docs": "/docs"  # Swagger UI
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    Why: Deployment platforms (Render, AWS) ping this to verify service is alive.
    """
    return {
        "status": "healthy",
        "debug_mode": settings.DEBUG
    }
