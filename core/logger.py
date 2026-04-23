"""
core/logger.py
==============
Centralised logging setup.
Usage in any module:
    from core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened", extra={"doc_id": "abc123"})
"""

import logging
import json
import sys
from datetime import datetime, timezone
from core.config import settings


# Reserved LogRecord attribute names
RESERVED_ATTRS = {
    'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
    'funcName', 'levelname', 'levelno', 'lineno', 'module', 'msecs',
    'message', 'msg', 'name', 'pathname', 'process', 'processName',
    'relativeCreated', 'stack_info', 'thread', 'threadName'
}


class _JsonFormatter(logging.Formatter):
    """Emits each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Attach any extra fields passed via extra={}
        # Skip reserved attributes to avoid KeyError
        for key, value in record.__dict__.items():
            if key not in RESERVED_ATTRS and not key.startswith('_'):
                # Handle non-serializable values
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)
        
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class _PlainFormatter(logging.Formatter):
    """Human-readable logs for development."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)


def _build_root_logger() -> None:
    """Configure the root logger once at import time."""
    root = logging.getLogger()

    # Avoid adding duplicate handlers if module is reloaded
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _JsonFormatter() if settings.log_json else _PlainFormatter()
    )
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "motor", "pymongo", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_build_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.
    Call this at the top of every module:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)