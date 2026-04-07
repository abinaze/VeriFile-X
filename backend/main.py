"""
VeriFile-X API - Privacy-preserving digital forensics platform.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from datetime import datetime
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import os

from backend.core.config import settings
from backend.core.logger import setup_logger
from backend.api.routes import upload, analyze, cases

logger = setup_logger(__name__)

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
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API routers
app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(cases.router)


@app.get("/")
async def root():
    """
    Serve frontend HTML.
    In production, this serves the web UI.
    """
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    else:
        return {
            "name": settings.API_TITLE,
            "version": settings.API_VERSION,
            "status": "operational",
            "docs": "/docs"
        }


@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "debug_mode": settings.DEBUG,
        "timestamp": datetime.now().isoformat()
    }
