"""Deterministic structured logging for Cosmic Foundry.

All log records produced by this package use ``StructuredFormatter``, which
serializes each record as a single JSON line.  The structure is deterministic:
the same operation always produces the same keys regardless of timing, so log
output is diff-friendly and machine-parseable.

Usage (in library code)::

    from cosmic_foundry.observability import get_logger
    log = get_logger(__name__)
    log.debug("dispatch.execute", extra={"region_shape": shape, "n_blocks": n})

Usage (in tests / applications)::

    from cosmic_foundry.observability import configure
    configure(level=logging.DEBUG)
"""

from __future__ import annotations

import json
import logging
from typing import Any

_STDLIB_FIELDS: frozenset[str] = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class StructuredFormatter(logging.Formatter):
    """Serializes each log record as a single JSON line.

    Fixed keys: ``level``, ``logger``, ``event``.
    Any extra fields passed via ``extra={...}`` are merged into the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STDLIB_FIELDS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a ``logging.Logger`` for *name*.

    The logger inherits handlers from the root logger.  Call
    :func:`configure` once (e.g. in tests or application entry points) to
    attach a :class:`StructuredFormatter` handler.
    """
    return logging.getLogger(name)


def configure(
    level: int = logging.WARNING,
    *,
    handler: logging.Handler | None = None,
) -> None:
    """Attach a :class:`StructuredFormatter` handler to the root logger.

    Safe to call multiple times; duplicate handlers are not added.  Intended
    for tests and CLI entry points, not library code.
    """
    root = logging.getLogger()
    root.setLevel(level)

    target = handler if handler is not None else logging.StreamHandler()
    formatter = StructuredFormatter()
    target.setFormatter(formatter)

    # Avoid duplicate handlers if configure() is called more than once.
    for existing in root.handlers:
        if type(existing) is type(target) and isinstance(
            existing.formatter, StructuredFormatter
        ):
            return
    root.addHandler(target)


__all__ = ["StructuredFormatter", "configure", "get_logger"]
