"""Structured logging for the entire pipeline.

Provides a logger that accepts structlog-style kwargs whether or not structlog
is installed. With stdlib logging, kwargs are flattened into the message.
"""
from __future__ import annotations
import logging
import sys
from typing import Any

try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False


class _StdlibKwargsAdapter:
    """Adapter so logger.info('event', key=value) works without structlog."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _format(self, event: str, **kwargs) -> str:
        if not kwargs:
            return event
        kv = " ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"{event} | {kv}" if event else kv

    def debug(self, event: str = "", **kwargs) -> None:
        self._logger.debug(self._format(event, **kwargs))

    def info(self, event: str = "", **kwargs) -> None:
        self._logger.info(self._format(event, **kwargs))

    def warning(self, event: str = "", **kwargs) -> None:
        self._logger.warning(self._format(event, **kwargs))

    def error(self, event: str = "", **kwargs) -> None:
        self._logger.error(self._format(event, **kwargs))

    def exception(self, event: str = "", **kwargs) -> None:
        self._logger.exception(self._format(event, **kwargs))


def configure_logging(level: str = "INFO") -> None:
    """Configure logging for the entire app."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    if _HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
        )


def get_logger(name: str) -> Any:
    """Return a kwargs-friendly logger."""
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return _StdlibKwargsAdapter(name)
