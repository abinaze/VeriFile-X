"""
VeriFile-X API - Privacy-preserving digital forensics platform.

Security features:
- Rate limiting (10 requests/minute per IP)
- CORS restricted to known origins
- No file storage (in-memory only)
- Input validation on all endpoints
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from backend.core.config import settings
from backend.core.logger import setup_logger
from backend.api.routes import upload, analyze

logger = setup_logger(__name__)

# Rate limiter - identifies clients by IP address
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle events."""
    logger.info("🚀 VeriFile-X API starting up...")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Max file size: {settings.MAX_FILE_SIZE_MB}MB")
    yield
    logger.info("🛑 VeriFile-X API shutting down...")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
)

# Attach rate limiter to app state
app.state.limiter = limiter

# Handle rate limit exceeded with proper JSON response
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Rate limiting middleware
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload.router)
app.include_router(analyze.router)


@app.get("/")
@limiter.limit("30/minute")
async def root(request: Request):
    """Root endpoint - API information."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """
    Health check endpoint for monitoring.
    Higher rate limit - used by uptime monitors.
    """
    return {
        "status": "healthy",
        "debug_mode": settings.DEBUG,
        "timestamp": datetime.now().isoformat()
    }
