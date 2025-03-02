"""
Logging configuration for the code assistant.

This module sets up logging for the application with different handlers
and formatters based on configuration.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .settings import get_settings


def configure_logging(config_file: Optional[Union[str, Path]] = None) -> None:
    """
    Configure logging for the application.
    
    Args:
        config_file: Path to a config file for settings
    """
    # Get settings
    settings = get_settings(config_file)
    
    # Get logging configuration
    log_level_str = settings.get(["logging", "level"], "INFO")
    log_file = settings.get(["logging", "file"], "codeassistant.log")
    log_format = settings.get(["logging", "format"], 
                              "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Convert log level string to level value
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    file_formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10 MB max, 5 backup files
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
    except Exception as e:
        # If file handler creation fails, just use console handler
        root_logger.addHandler(console_handler)
        root_logger.warning(f"Could not create file handler: {str(e)}")
        
    # Set specific levels for noisy modules
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Confirmation message
    root_logger.info(f"Logging configured with level {log_level_str}")
    

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Name of the module
        
    Returns:
        Logger instance for the module
    """
    return logging.getLogger(name)