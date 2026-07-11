"""
Centralized logging configuration.
"""
import logging
import sys
from backend.core.config import settings


def setup_logger(name: str) -> logging.Logger:
    """Create logger with consistent formatting."""
    logger = logging.getLogger(name)

    # BUG FIX: previously ignored settings.LOG_LEVEL entirely (declared in
    # .env.example as a user-configurable knob, defaulting to "INFO", but
    # never read here) — level was derived only from settings.DEBUG. DEBUG
    # still forces DEBUG level for convenience; otherwise LOG_LEVEL is now
    # honored, falling back to INFO if it's unset or not a recognized name.
    if settings.DEBUG:
        level = logging.DEBUG
    else:
        level = getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger
