"""Centralized logging configuration for Clockify RAG system.

This module provides a single point of configuration for all logging
across the application, ensuring consistent formatting, levels, and handlers.
"""

import logging
import sys
from typing import Optional
from pathlib import Path
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_obj["extra"] = record.extra

        return json.dumps(log_obj)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter with colors (if supported)."""

    # Color codes (ANSI)
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self, use_colors: bool = True):
        """Initialize formatter.

        Args:
            use_colors: Whether to use ANSI color codes
        """
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors."""
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            reset = self.COLORS["RESET"]
            record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


def setup_logging(
    level: str = "INFO",
    format_type: str = "text",
    log_file: Optional[str] = None,
    use_colors: bool = True,
    quiet: bool = False,
) -> None:
    """Central logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ("text" or "json")
        log_file: Optional file path for file logging
        use_colors: Use colored output for console (text mode only)
        quiet: Suppress console output (only log to file)

    Example:
        >>> # Development mode
        >>> setup_logging(level="DEBUG", format_type="text", use_colors=True)
        >>>
        >>> # Production mode
        >>> setup_logging(level="INFO", format_type="json", log_file="/var/log/rag.log")
        >>>
        >>> # File-only logging
        >>> setup_logging(level="INFO", log_file="app.log", quiet=True)
    """
    # Get root logger
    root = logging.getLogger()

    # Clear any existing handlers to avoid duplicates
    root.handlers.clear()

    # Set level
    try:
        log_level = getattr(logging, level.upper())
    except AttributeError:
        log_level = logging.INFO
        print(f"Warning: Invalid log level '{level}', using INFO", file=sys.stderr)

    root.setLevel(log_level)

    # Create formatter based on type
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter(use_colors=use_colors)

    # Console handler (unless quiet mode)
    if not quiet:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)
        console.setFormatter(formatter)
        root.addHandler(console)

    # File handler (if specified)
    if log_file:
        # Create parent directories if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use JSON format for file logs (easier to parse)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)

        # Always use JSON for file logs for easier parsing
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)

    # Set third-party library log levels to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Log configuration applied
    root.info(
        f"Logging configured: level={level}, format={format_type}, "
        f"file={log_file or 'none'}, quiet={quiet}"
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Logger name (usually __name__ of the module)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return logging.getLogger(name)


# Convenience function for testing
def reset_logging() -> None:
    """Reset logging configuration.

    Useful for testing to ensure clean state between tests.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
