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
import random as _random
import numpy as _np_seed

# Deterministic seeds — critical for a forensics tool where the same
# image must always produce the same result across server restarts.
# Note: MCMC uses an image-derived seed (correct); these seeds cover
# the global numpy RNG used by KL divergence and Mahalanobis distance.
_random.seed(42)
_np_seed.random.seed(42)
try:
    import torch as _torch_seed
    _torch_seed.manual_seed(42)
    if _torch_seed.cuda.is_available():
        _torch_seed.cuda.manual_seed_all(42)
except ImportError:
    pass

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

    # ── Download model files from HF Space repo if missing ────────────────
    # On HuggingFace Spaces, Git LFS files are not materialised during the
    # Docker build (COPY . . copies pointer stubs instead of real binaries).
    # We fetch them at startup via the Hub HTTP API instead.
    _ref = Path(__file__).parent.parent / "data" / "reference"
    _ref.mkdir(parents=True, exist_ok=True)
    _model_files = [
        "own_embedding_model.pt",
        "clip_database.pkl",
        "own_centroids.pkl",
        "ensemble_xgb.pkl",
        "ensemble_results.json",
    ]
    _space_id = os.environ.get("SPACE_ID", "abinazebinoy/verifile-x-api")
    _need_download = any(
        not (_ref / f).exists() or (_ref / f).stat().st_size < 1000
        for f in _model_files
    )
    if _need_download:
        logger.info(f"Downloading model files from HuggingFace Space: {_space_id}")
        try:
            from huggingface_hub import hf_hub_download
            for _fname in _model_files:
                _dest = _ref / _fname
                if _dest.exists() and _dest.stat().st_size > 1000:
                    logger.info(f"  Already present: {_fname}")
                    continue
                try:
                    _path = hf_hub_download(
                        repo_id=_space_id,
                        filename=f"data/reference/{_fname}",
                        repo_type="space",
                        local_dir=str(Path(__file__).parent.parent),
                    )
                    logger.info(f"  Downloaded: {_fname} ({Path(_path).stat().st_size // 1024} KB)")
                except Exception as _e:
                    logger.warning(f"  Could not download {_fname}: {_e}")
        except ImportError:
            logger.warning("huggingface_hub not installed — cannot auto-download model files")
    else:
        logger.info("All model files present on disk.")

    # ── Register HEIF/HEIC format support via pillow-heif ─────────────────
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
    from fastapi import HTTPException  # import once at top of function scope
    import hashlib as _hl
    import os as _os
    admin_key = request.headers.get("X-Admin-Key", "")
    if not admin_key or len(admin_key) < 16:
        raise HTTPException(status_code=401, detail="X-Admin-Key header required")
    # Compare against SHA-256 hash stored in ADMIN_KEY_HASH env var.
    # If ADMIN_KEY_HASH is not set, fall back to length check only (dev mode).
    expected_hash = _os.getenv("ADMIN_KEY_HASH", "")
    if not expected_hash:
        # Only hard-block when explicitly deployed to production.
        # Production is indicated by PRODUCTION=true env var.
        # DEBUG=true, CI=true, or unset env vars all allow length-only check
        # (dev, CI, and staging environments should not require ADMIN_KEY_HASH).
        _is_production = _os.getenv("PRODUCTION", "").lower() in ("1", "true", "yes")
        if _is_production:
            logger.error("ADMIN_KEY_HASH not set in production — admin access blocked.")
            raise HTTPException(
                status_code=503,
                detail="Admin access is disabled: ADMIN_KEY_HASH environment variable is not configured."
            )
        logger.warning(
            "ADMIN_KEY_HASH not set — using length-only check. "
            "Set ADMIN_KEY_HASH and PRODUCTION=true before deploying to production."
        )
    if expected_hash:
        provided_hash = _hl.sha256(admin_key.encode()).hexdigest()
        if not secrets.compare_digest(provided_hash, expected_hash):
            raise HTTPException(status_code=401, detail="X-Admin-Key header required")
    from backend.services.metrics_collector import reset_metrics
    reset_metrics()
    return {"message": "Metrics reset successfully."}

@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Reports real detector model loading status — not just process liveness."""
    from pathlib import Path as _HPath
    _ref = _HPath(__file__).parent.parent / "data" / "reference"
    _clip_ok  = (_ref / "clip_database.pkl").exists()
    _own_ok   = (_ref / "own_embedding_model.pt").exists()
    _xgb_ok   = (_ref / "ensemble_xgb.pkl").exists()
    _platt_ok = (_ref / "platt_params.json").exists()
    _degraded = not (_clip_ok and _own_ok and _xgb_ok)
    return {
        "status": "degraded" if _degraded else "healthy",
        "debug_mode": settings.DEBUG,
        "timestamp": datetime.now().isoformat(),
        "detector_models": {
            "clip_database":     "ok" if _clip_ok  else "missing",
            "own_embedding":     "ok" if _own_ok   else "missing",
            "xgboost_ensemble":  "ok" if _xgb_ok   else "missing",
            "platt_calibration": "ok" if _platt_ok else "defaults",
        },
        "accuracy_note": (
            "Running on statistical signals only (~55-68% accuracy). "
            "Build reference models for full accuracy." if _degraded
            else "All models loaded."
        ),
    }
