"""
Centralized logging configuration.
"""
import logging
import sys
from backend.core.config import settings


def setup_logger(name: str) -> logging.Logger:
    """Create logger with consistent formatting."""
    logger = logging.getLogger(name)
    
    level = logging.DEBUG if settings.DEBUG else logging.INFO
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
