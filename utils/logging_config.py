# -*- coding: utf-8 -*-
import logging
import os
import sys
import config # Assuming config is importable from project root

def setup_logging():
    """Configures logging to output to both console and file based on config."""
    log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    formatter = logging.Formatter(config.LOG_FORMAT)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        file_handler = logging.FileHandler(config.LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        # Use root logger to print this message to ensure it goes to console and file
        root_logger.info(f"Logging configured. Level: {config.LOG_LEVEL}, File: {config.LOG_FILE}")
    except Exception as e:
        # Use root logger here as well
        root_logger.error(f"Failed to set up file logging to {config.LOG_FILE}: {e}", exc_info=True)

__all__ = ["setup_logging"] 