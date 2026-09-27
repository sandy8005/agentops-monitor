"""
Centralized structured logging (stdlib, no new deps).

Every core module gets a logger via get_logger(__name__). Logs carry structured
context — run_id and step where available — via `extra=`, and the formatter emits
a consistent, greppable line:

    2026-01-01 12:00:00 INFO  router  parsed resume from cache  [run=42 step=7]

Scope: configuration is applied ONLY to this application's logger namespace
("agentops.*"). The ROOT logger and any handlers the host installed (Uvicorn,
Gunicorn, a platform log shipper) are left untouched — the previous version called
root.handlers.clear(), which silently removed the hosting environment's handlers.
App loggers do not propagate to the root, so a host that also logs at the root
level doesn't print every line twice.

Level comes from settings.log_level (LOG_LEVEL, default INFO), the single source of
configuration. Because runs execute in the worker process (not the API), both
api.py and worker.py import this so their logs look the same.

Privacy: log METADATA (ids, counts, statuses), never resume- or job-derived
content (advice text, prompts, responses). Application logs are not covered by the
trace-retention purge — see README "Data handling".

Usage:
    from logging_config import get_logger
    log = get_logger(__name__)
    log.info("searching jobs", extra={"run_id": run_id, "step_id": step_id})
"""
import logging
import sys

APP_LOGGER = "agentops"
_CONFIGURED = False


class _ContextFormatter(logging.Formatter):
    """Formatter that appends run=/step= when those keys are in the record's extra,
    and shows the module name without the 'agentops.' namespace prefix."""

    def format(self, record):
        short = record.name[len(APP_LOGGER) + 1:] if record.name.startswith(APP_LOGGER + ".") \
            else record.name
        original, record.name = record.name, short
        try:
            base = super().format(record)
        finally:
            record.name = original
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
    """Set up the app's logger namespace once. Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    if level is None:
        try:
            from settings import settings
            level = settings.log_level
        except Exception:
            level = "INFO"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ContextFormatter(
        fmt="%(asctime)s %(levelname)-5s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    app = logging.getLogger(APP_LOGGER)
    # Replace only OUR previously-installed handler(s), never anyone else's.
    for h in list(app.handlers):
        if getattr(h, "_agentops_handler", False):
            app.removeHandler(h)
    handler._agentops_handler = True
    app.addHandler(handler)
    app.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    app.propagate = False
    # Quiet noisy third-party loggers (level only — their handlers are not touched).
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name):
    """Return a module logger inside the app namespace, ensuring logging is configured."""
    configure()
    return logging.getLogger(f"{APP_LOGGER}.{name.split('.')[-1]}")


# Configure on import so a bare `from logging_config import get_logger` just works.
configure()