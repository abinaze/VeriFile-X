"""
Centralized logging configuration.
Why: Consistent log format, easy to change output (file vs console).
"""
import logging
import sys
from backend.core.config import settings

def setup_logger(name: str) -> logging.Logger:
    """
    Create a logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__ of the module)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level based on DEBUG mode
    level = logging.DEBUG if settings.DEBUG else logging.INFO
    logger.setLevel(level)
    
    # Prevent duplicate handlers if already configured
    if logger.handlers:
        return logger
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Format: timestamp - logger_name - level - message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger
