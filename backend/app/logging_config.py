"""Centralized terminal logging configuration for Parcel Intelligence.

Provides colored, high-visibility log and trace output in the terminal for:
- Orchestrator pipeline steps (ingestion, 3-branch execution, evidence reconciliation)
- Background worker queue operations (enqueue, worker pickup, job completion)
- Machine metadata OCR, physical label barcode decoding, and zero-shot detector results
"""

import logging
import os
import sys
from typing import Optional

from app.config import settings


class TerminalColorFormatter(logging.Formatter):
    """
    Formatter supporting ANSI color formatting for terminal logs.
    Gracefully degrades to plain text when output is not a terminal.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def __init__(self, use_colors: Optional[bool] = None):
        super().__init__()
        if use_colors is None:
            # Enable colors if running in a TTY or forced via environment
            self.use_colors = sys.stdout.isatty() or os.getenv("FORCE_COLOR") == "1"
        else:
            self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        time_str = self.formatTime(record, "%H:%M:%S")

        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            level_str = f"{color}{record.levelname:<7}{self.RESET}"
            name_str = f"{self.BOLD}[{record.name:<13}]{self.RESET}"
            msg_str = record.getMessage()
            if record.exc_info:
                # Add formatted exception trace
                if not record.exc_text:
                    record.exc_text = self.formatException(record.exc_info)
                if record.exc_text:
                    msg_str = f"{msg_str}\n{self.COLORS['ERROR']}{record.exc_text}{self.RESET}"
            return f"{self.DIM}{time_str}{self.RESET} {level_str} {name_str} {msg_str}"
        else:
            level_str = f"{record.levelname:<7}"
            name_str = f"[{record.name:<13}]"
            msg = record.getMessage()
            if record.exc_info:
                if not record.exc_text:
                    record.exc_text = self.formatException(record.exc_info)
                if record.exc_text:
                    msg = f"{msg}\n{record.exc_text}"
            return f"{time_str} {level_str} {name_str} {msg}"


_logging_initialized = False


def setup_logging(level: str | None = None) -> None:
    """
    Initialize terminal logging handlers and levels.
    Idempotent: safe to call multiple times.
    """
    global _logging_initialized
    if _logging_initialized:
        return

    log_level_name = (level or getattr(settings, "LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Check if a custom stream handler is already configured
    has_terminal_handler = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "_is_parcel_handler", False)
        for h in root_logger.handlers
    )

    if not has_terminal_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(TerminalColorFormatter())
        handler._is_parcel_handler = True
        root_logger.addHandler(handler)

    # Ensure application loggers propagate at INFO level
    app_loggers = [
        "orchestrator",
        "worker_queue",
        "preprocessor",
        "detector",
        "overlay_extractor",
        "ocr",
        "barcode",
        "parser",
        "parser_registry",
        "s3_service",
        "api",
    ]
    for name in app_loggers:
        lg = logging.getLogger(name)
        lg.setLevel(log_level)
        lg.propagate = True

    # Silence noisy 3rd-party loggers
    noisy_loggers = [
        "botocore",
        "urllib3",
        "s3transfer",
        "filelock",
        "transformers",
        "PIL",
        "asyncio",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    _logging_initialized = True
