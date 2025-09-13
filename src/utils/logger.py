"""
Logging utilities for CrisisMapper.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger


def setup_logger(name: str, 
                 level: str = "INFO",
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Setup logger for CrisisMapper modules.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_string: Optional custom format string
        
    Returns:
        Configured logger
    """
    # Remove default loguru handler
    loguru_logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    loguru_logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        loguru_logger.add(
            log_file_path,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
    
    # Create standard logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Add loguru handler to standard logger
    handler = InterceptHandler()
    logger.addHandler(handler)
    
    return logger


class InterceptHandler(logging.Handler):
    """Handler to intercept standard logging and redirect to loguru."""
    
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller from where the logged message originated
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for a module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
