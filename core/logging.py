"""Structured logging setup. Call `configure_logging()` once at each entrypoint
(scripts/*.py, app/backend/main.py, app/frontend/Home.py); call `get_logger(__name__)`
everywhere else.
"""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Idempotent — safe to call from multiple entrypoints or in tests."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = (level or os.environ.get("PLANOGRID_LOG_LEVEL", "INFO")).upper()

    logging.basicConfig(
        level=resolved_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
