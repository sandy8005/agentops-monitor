"""
Centralized structured logging (stdlib, no new deps).

Every core module gets a logger via get_logger(__name__). Logs carry structured
context — run_id and step where available — via `extra=`, and the formatter emits
a consistent, greppable line:

    2026-01-01 12:00:00 INFO  router  parsed resume from cache  [run=42 step=7]

Configuration is applied ONCE (configure() is idempotent) — importing this module
sets it up. Level comes from the LOG_LEVEL env var (default INFO). Because runs
execute in the worker process (not the API), both api.py and worker.py import and
configure this so their logs look the same.

Usage:
    from logging_config import get_logger
    log = get_logger(__name__)
    log.info("searching jobs", extra={"run_id": run_id, "step_id": step_id})
    log.warning("adzuna rate limited", extra={"run_id": run_id})
"""
import logging
import os
import sys

_CONFIGURED = False


class _ContextFormatter(logging.Formatter):
    """Formatter that appends run=/step= when those keys are in the record's extra."""

    def format(self, record):
        base = super().format(record)
        ctx = []
        run_id = getattr(record, "run_id", None)
        step_id = getattr(record, "step_id", None)
        if run_id is not None:
            ctx.append(f"run={run_id}")
        if step_id is not None:
            ctx.append(f"step={step_id}")
        if ctx:
            return f"{base}  [{' '.join(ctx)}]"
        return base


def configure(level=None):
    """Set up root logging once. Safe to call repeatedly (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ContextFormatter(
        fmt="%(asctime)s %(levelname)-5s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))
    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name):
    """Return a module logger, ensuring logging is configured."""
    configure()
    return logging.getLogger(name.split(".")[-1])


# Configure on import so a bare `from logging_config import get_logger` just works.
configure()