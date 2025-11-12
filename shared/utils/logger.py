"""
Shared Logging Utilities
Provides consistent logging across all agents
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """Format logs as JSON for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    log_dir: str = "/logs"
) -> logging.Logger:
    """
    Setup logger with consistent configuration

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        json_format: Use JSON formatting
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)

    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_dir) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))

        if json_format:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)

        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create new one with default settings

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Setup with defaults if not already configured
        return setup_logger(name)

    return logger
