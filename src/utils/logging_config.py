import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


# Default log format with timestamp, level, logger name, and message
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
# Simpler format for console output
CONSOLE_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

# Log file path
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "error.log")


def setup_logging(
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_format: Optional[str] = None,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to LOG_LEVEL env var or "INFO".
        log_to_file: Whether to write logs to files (default: True).
        log_format: Custom log format string. If None, uses default.
    """
    # Determine log level
    log_level_str = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Determine log format
    fmt = log_format or CONSOLE_LOG_FORMAT

    # Create root logger for the application
    root_logger = logging.getLogger("src")
    root_logger.setLevel(log_level)

    # Prevent adding duplicate handlers if setup_logging is called multiple times
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Console handler - always enabled
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(fmt))
    root_logger.addHandler(console_handler)

    # File handlers - optional
    if log_to_file:
        # Create logs directory if it doesn't exist
        os.makedirs(LOG_DIR, exist_ok=True)

        # General log file (all levels)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        root_logger.addHandler(file_handler)

        # Error log file (errors and above only)
        error_handler = RotatingFileHandler(
            ERROR_LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        root_logger.addHandler(error_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("pgvector").setLevel(logging.WARNING)

    # Log startup message
    root_logger.info("=" * 80)
    root_logger.info("Application logging initialized")
    root_logger.info(f"Log level: {log_level_str}")
    root_logger.info(f"Log to file: {log_to_file}")
    root_logger.info(f"Log directory: {LOG_DIR if log_to_file else 'disabled'}")
    root_logger.info("=" * 80)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
