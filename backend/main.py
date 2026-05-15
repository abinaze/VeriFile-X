from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import secrets
from datetime import datetime
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import os

from backend.core.config import settings
from backend.core.logger import setup_logger
from backend.api.routes import upload, analyze, cases, keys
from backend.api.routes import webhooks
from backend.api.routes import feedback

logger = setup_logger(__name__)

# Shared rate limiter — imported by all routes
limiter = Limiter(key_func=get_remote_address)
shared_limiter = limiter  # alias for explicit import


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VeriFile-X API starting up")

    # Register HEIF/HEIC format support via pillow-heif.
    # Must be called before any Pillow image operations.
    # Falls back silently if pillow-heif is not installed.
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        logger.info("pillow-heif registered: HEIC/HEIF format support enabled")
    except ImportError:
        logger.warning(
            "pillow-heif not installed — HEIC/HEIF images will be rejected. "
            "Install with: pip install pillow-heif"
        )
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Max file size: {settings.MAX_FILE_SIZE_MB}MB")
    yield
    logger.info("VeriFile-X API shutting down")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["X-XSS-Protection"]       = "1; mode=block"
    response.headers["Referrer-Policy"]         = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"]           = "no-store" if "/api/" in str(request.url.path) else "public, max-age=3600"
    # HSTS must only be sent over HTTPS — sending it on HTTP is invalid (RFC 6797)
    if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"]   = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data: https://raw.githubusercontent.com"
    response.headers["Permissions-Policy"]        = "camera=(), microphone=(), geolocation=()"
    return response

app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(cases.router)
app.include_router(keys.router)
app.include_router(webhooks.router)
app.include_router(feedback.router)


@app.get("/")
async def root():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "docs": "/docs",
    }




@app.get("/api/v1/metrics", tags=["Observability"], summary="System metrics")
@limiter.limit("30/minute")
async def get_metrics(request: Request):
    """Return real-time system metrics: request rates, score distributions, latency."""
    from backend.services.metrics_collector import get_metrics
    return get_metrics()


@app.post("/api/v1/metrics/reset", tags=["Observability"], summary="Reset metrics (admin)")
@limiter.limit("5/minute")
async def reset_metrics_endpoint(request: Request):
    """Reset all metrics counters. Requires X-Admin-Key header."""
    import hashlib as _hl
    import os as _os
    admin_key = request.headers.get("X-Admin-Key", "")
    if not admin_key or len(admin_key) < 16:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="X-Admin-Key header required")
    # Compare against SHA-256 hash stored in ADMIN_KEY_HASH env var.
    # If ADMIN_KEY_HASH is not set, fall back to length check only (dev mode).
    expected_hash = _os.getenv("ADMIN_KEY_HASH", "")
    if not expected_hash:
        logger.warning(
            "ADMIN_KEY_HASH env var not set. Any string >=16 chars grants admin access. "
            "Set ADMIN_KEY_HASH in production (sha256 of your key) to enforce proper auth."
        )
    if expected_hash:
        provided_hash = _hl.sha256(admin_key.encode()).hexdigest()
        if not secrets.compare_digest(provided_hash, expected_hash):
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail="X-Admin-Key header required")
    from backend.services.metrics_collector import reset_metrics
    reset_metrics()
    return {"message": "Metrics reset successfully."}

@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    return {
        "status": "healthy",
        "debug_mode": settings.DEBUG,
        "timestamp": datetime.now().isoformat(),
    }
