"""Emit coarse Sphinx phase timings in CI logs."""

from __future__ import annotations

import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

_PHASES = (
    "read_environment",
    "read_docs",
    "read_source_and_doctree",
    "write",
    "total_build",
)


def _state(app: Sphinx) -> SimpleNamespace:
    state = getattr(app, "_cosmic_foundry_timing", None)
    if state is None:
        state = SimpleNamespace(
            started_at=time.perf_counter(),
            started={},
            totals=defaultdict(float),
            counts=defaultdict(int),
        )
        app._cosmic_foundry_timing = state
    return state


def _start(app: Sphinx, phase: str) -> None:
    _state(app).started[phase] = time.perf_counter()


def _stop(app: Sphinx, phase: str) -> None:
    state = _state(app)
    started_at = state.started.pop(phase, None)
    if started_at is None:
        return
    state.totals[phase] += time.perf_counter() - started_at
    state.counts[phase] += 1


def _on_builder_inited(app: Sphinx) -> None:
    _start(app, "read_environment")
    logger.info("[docs-timing] builder-inited")


def _on_env_before_read_docs(app: Sphinx, env: Any, docnames: set[str]) -> None:
    state = _state(app)
    _start(app, "read_docs")
    state.counts["scheduled_docs"] = len(docnames)
    logger.info("[docs-timing] scheduled documents %d", len(docnames))


def _on_source_read(app: Sphinx, docname: str, source: list[str]) -> None:
    _state(app).started[("source", docname)] = time.perf_counter()


def _on_doctree_read(app: Sphinx, doctree: Any) -> None:
    state = _state(app)
    docname = app.env.current_document.docname
    started_at = state.started.pop(("source", docname), None)
    if started_at is None:
        return
    state.totals["read_source_and_doctree"] += time.perf_counter() - started_at
    state.counts["read_source_and_doctree"] += 1


def _on_env_updated(app: Sphinx, env: Any) -> None:
    _stop(app, "read_docs")
    _stop(app, "read_environment")
    logger.info("[docs-timing] read_docs %.3fs", _state(app).totals["read_docs"])


def _on_doctree_resolved(app: Sphinx, doctree: Any, docname: str) -> None:
    _state(app).counts["resolved_doctrees"] += 1


def _on_write_started(app: Sphinx, builder: Any) -> None:
    _start(app, "write")


def _on_build_finished(app: Sphinx, exception: Exception | None) -> None:
    state = _state(app)
    _stop(app, "write")
    state.totals["total_build"] = time.perf_counter() - state.started_at

    logger.info("")
    logger.info("====================== Sphinx phase timings ======================")
    for phase in _PHASES:
        if phase in state.totals:
            count = state.counts.get(phase)
            suffix = f" ({count})" if count else ""
            logger.info("[docs-timing] %s %.3fs%s", phase, state.totals[phase], suffix)
    for count in ("scheduled_docs", "resolved_doctrees"):
        if count in state.counts:
            logger.info("[docs-timing] %s %d", count, state.counts[count])
    logger.info("")


def setup(app: Sphinx) -> dict[str, bool | str]:
    app.connect("builder-inited", _on_builder_inited)
    app.connect("env-before-read-docs", _on_env_before_read_docs)
    app.connect("source-read", _on_source_read)
    app.connect("doctree-read", _on_doctree_read)
    app.connect("env-updated", _on_env_updated)
    app.connect("doctree-resolved", _on_doctree_resolved)
    app.connect("write-started", _on_write_started)
    app.connect("build-finished", _on_build_finished)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
