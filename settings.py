"""
Centralized application settings.

Loads the .env file ONCE (on import) and exposes all configuration through a
single typed `settings` object, so no other module needs to call os.getenv or
load_dotenv itself. This is the one place that knows how config is sourced —
including LOG_LEVEL (logging_config reads settings.log_level).

Usage:
    from settings import settings
    conn_kwargs = settings.db_kwargs()
    if settings.redact_trace_payloads: ...
    key = settings.gemini_api_key

Validation: every value is parsed and range-checked at construction. A bad value
(DB_POOL_MAX=abc, SESSION_MAX_AGE=0, DB_POOL_MAX < DB_POOL_MIN, ...) raises ONE
SettingsError that lists every problem by variable name — never a bare ValueError
from deep inside int().

Required-vs-optional: DB_* are required to talk to Postgres; that check runs when
the DB pool is first created (validate_db, not at import), so pure-logic code and
tests can import freely without a live DB config. API keys are optional (features
degrade without them). SESSION_SECRET is required only when ENV=production
(enforced in api.py, where a forgeable session key actually matters).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# LangGraph reads LANGGRAPH_STRICT_MSGPACK when its serde module is imported. Make the
# SAFE setting the default for this app, before anything imports langgraph. An
# explicit value in the environment/.env still wins (setdefault). The checkpointer is
# ALSO built with an explicit strict serializer (see checkpointing.make_serde),
# so this is defense in depth rather than the only control.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")


class SettingsError(ValueError):
    """One or more environment variables are invalid."""


_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


def _get(name, default=None):
    v = os.getenv(name)
    return v if v not in (None, "") else default


class _Reader:
    """Collects parse errors instead of raising on the first one."""

    def __init__(self):
        self.errors = []

    def str(self, name, default=None):
        return _get(name, default)

    def bool(self, name, default=False):
        v = os.getenv(name)
        if v is None or v.strip() == "":
            return default
        low = v.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        self.errors.append(f"{name}={v!r} is not a boolean (use true/false)")
        return default

    def int(self, name, default, min_value=None, max_value=None):
        raw = _get(name)
        if raw is None:
            val = default
        else:
            try:
                val = int(raw.strip())
            except ValueError:
                self.errors.append(f"{name}={raw!r} is not an integer")
                return default
        if min_value is not None and val < min_value:
            self.errors.append(f"{name}={val} must be >= {min_value}")
        if max_value is not None and val > max_value:
            self.errors.append(f"{name}={val} must be <= {max_value}")
        return val

    def float(self, name, default=None, min_value=None, max_value=None):
        raw = _get(name)
        if raw is None:
            return default
        try:
            val = float(raw.strip())
        except ValueError:
            self.errors.append(f"{name}={raw!r} is not a number")
            return default
        if min_value is not None and val < min_value:
            self.errors.append(f"{name}={val} must be >= {min_value}")
        if max_value is not None and val > max_value:
            self.errors.append(f"{name}={val} must be <= {max_value}")
        return val


class Settings:
    """Typed, validated, read-once view of the environment. Instantiated once as
    `settings`."""

    def __init__(self):
        r = _Reader()

        # --- database (required for DB use) ---
        self.db_name = r.str("DB_NAME")
        self.db_user = r.str("DB_USER")
        self.db_password = r.str("DB_PASSWORD")
        self.db_host = r.str("DB_HOST")
        self.db_port = r.str("DB_PORT")
        # Operational timeouts for EVERY connection this app opens (pool, dedicated
        # run-lock connections, the LangGraph checkpointer). Without them a dead
        # network path can hang a worker indefinitely instead of failing fast.
        self.db_connect_timeout = r.int("DB_CONNECT_TIMEOUT", 10, min_value=1)
        self.db_keepalives_idle = r.int("DB_KEEPALIVES_IDLE", 30, min_value=1)
        self.db_keepalives_interval = r.int("DB_KEEPALIVES_INTERVAL", 10, min_value=1)
        self.db_keepalives_count = r.int("DB_KEEPALIVES_COUNT", 5, min_value=1)

        # DB connection pool bounds. Each PROCESS (api, worker) gets its own pool —
        # see README "Connection budget" for the full per-process arithmetic.
        self.db_pool_min = r.int("DB_POOL_MIN", 1, min_value=1)
        self.db_pool_max = r.int("DB_POOL_MAX", 20, min_value=1)
        if self.db_pool_max < self.db_pool_min:
            r.errors.append(f"DB_POOL_MAX={self.db_pool_max} must be >= "
                            f"DB_POOL_MIN={self.db_pool_min}")

        # --- environment / deploy ---
        self.env = (r.str("ENV", "dev") or "dev").lower()
        self.session_secret = r.str("SESSION_SECRET")
        self.session_max_age = r.int("SESSION_MAX_AGE", 8 * 60 * 60, min_value=1)
        # What redaction covers is spelled out in api.py (_redact_* helpers) and the
        # README. REDACT_TRACE_PAYLOADS is the accurate name; REDACT_SENSITIVE is the
        # deprecated alias, honoured when the new name is not set.
        if _get("REDACT_TRACE_PAYLOADS") is not None:
            self.redact_trace_payloads = r.bool("REDACT_TRACE_PAYLOADS", True)
        else:
            self.redact_trace_payloads = r.bool("REDACT_SENSITIVE", True)
        self.log_level = (r.str("LOG_LEVEL", "INFO") or "INFO").upper()
        if self.log_level not in _LOG_LEVELS:
            r.errors.append(f"LOG_LEVEL={self.log_level!r} must be one of "
                            f"{sorted(_LOG_LEVELS)}")

        # Rate limits (requests per window) — see api.py. The storage URI makes the
        # limits SHARED across API processes (e.g. redis://host:6379); the default
        # memory:// storage is per-process, so N processes allow N x the limit.
        self.rate_limit_login = r.str("RATE_LIMIT_LOGIN", "5/minute")
        self.rate_limit_runs = r.str("RATE_LIMIT_RUNS", "20/minute")
        self.rate_limit_storage_uri = r.str("RATE_LIMIT_STORAGE_URI", "memory://")

        # Trace retention: sensitive payloads of runs that ended more than this many
        # days ago are purged by the worker (metrics kept). 0 disables purging.
        self.trace_retention_days = r.int("TRACE_RETENTION_DAYS", 30, min_value=0)

        # Evaluator sampling for UNFLAGGED decisions when a run asks for evaluation:
        # flagged decisions are always evaluated; this fraction of the rest is too.
        self.eval_unflagged_sample_rate = r.float("EVAL_UNFLAGGED_SAMPLE_RATE", 1.0,
                                                  min_value=0.0, max_value=1.0)

        # --- LLM ---
        self.gemini_api_key = r.str("GEMINI_API_KEY")
        self.gemini_model = r.str("GEMINI_MODEL", "gemini-3.6-flash")
        # After the provider reports an exhausted quota, skip LLM calls in this worker
        # for this long (llm.py quota circuit breaker).
        self.llm_quota_cooldown_seconds = r.int("LLM_QUOTA_COOLDOWN_SECONDS", 3600,
                                                min_value=60, max_value=86400)
        # Optional explicit prices (USD per 1M tokens). Both must be set to apply.
        self.llm_input_price_per_million = r.float("LLM_INPUT_PRICE_PER_MILLION",
                                                   None, min_value=0.0)
        self.llm_output_price_per_million = r.float("LLM_OUTPUT_PRICE_PER_MILLION",
                                                    None, min_value=0.0)

        # --- LangGraph checkpoint hardening (see os.environ.setdefault above) ---
        self.langgraph_strict_msgpack = r.bool("LANGGRAPH_STRICT_MSGPACK", True)

        # --- Adzuna (optional live-jobs feed) ---
        self.adzuna_app_id = r.str("ADZUNA_APP_ID")
        self.adzuna_app_key = r.str("ADZUNA_APP_KEY")
        self.adzuna_country = r.str("ADZUNA_COUNTRY", "us")

        if r.errors:
            raise SettingsError("Invalid configuration:\n  - " + "\n  - ".join(r.errors))

    # Deprecated alias kept so older code/tests reading settings.redact_sensitive work.
    @property
    def redact_sensitive(self):
        return self.redact_trace_payloads

    # --- derived / helpers ---
    @property
    def is_production(self):
        return self.env in ("prod", "production")

    def db_kwargs(self):
        """The kwargs for psycopg2.connect / psycopg make_conninfo — the single
        definition of DB connection parameters, including operational timeouts and
        TCP keepalives, used by database.py, run_lock.py and the checkpointer."""
        return {
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
            "host": self.db_host,
            "port": self.db_port,
            "connect_timeout": self.db_connect_timeout,
            "keepalives": 1,
            "keepalives_idle": self.db_keepalives_idle,
            "keepalives_interval": self.db_keepalives_interval,
            "keepalives_count": self.db_keepalives_count,
        }

    def validate_db(self):
        """Fail loudly if required DB settings are missing. Called by the DB pool on
        first use so a misconfigured environment errors clearly instead of deep in a
        query — but NOT at import, so imports never require a live DB."""
        missing = [k for k in ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")
                   if not getattr(self, "db_" + k[3:].lower())]
        if missing:
            raise RuntimeError(
                "Missing required database settings in environment/.env: "
                + ", ".join(missing)
            )


settings = Settings()