"""
Centralized logging configuration.
"""
import logging
import sys
from backend.core.config import settings


def setup_logger(name: str) -> logging.Logger:
    """Create logger with consistent formatting."""
    logger = logging.getLogger(name)

    # DEBUG forces DEBUG level; otherwise settings.LOG_LEVEL is honored,
    # falling back to INFO if unset or not a recognized name.
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
