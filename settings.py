"""
Centralized application settings.

Loads the .env file ONCE (on import) and exposes all configuration through a
single typed `settings` object, so no other module needs to call os.getenv or
load_dotenv itself. This is the one place that knows how config is sourced.

Usage:
    from settings import settings
    conn_kwargs = settings.db_kwargs()
    if settings.redact_sensitive: ...
    key = settings.gemini_api_key

Required-vs-optional: DB_* are required to talk to Postgres; validation runs when
the DB pool is first created (not at import), so pure-logic code and tests can
import freely without a live DB config. API keys are optional (features degrade
without them). SESSION_SECRET is required only when ENV=production (enforced in
api.py, where a forgeable session key actually matters).
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _get(name, default=None):
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _get_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    """Typed, read-once view of the environment. Instantiated once as `settings`."""

    def __init__(self):
        # --- database (required) ---
        self.db_name = _get("DB_NAME")
        self.db_user = _get("DB_USER")
        self.db_password = _get("DB_PASSWORD")
        self.db_host = _get("DB_HOST")
        self.db_port = _get("DB_PORT")

        # --- environment / deploy ---
        self.env = (_get("ENV", "dev") or "dev").lower()
        self.session_secret = _get("SESSION_SECRET")
        self.redact_sensitive = _get_bool("REDACT_SENSITIVE", True)
        self.session_max_age = int(_get("SESSION_MAX_AGE", str(8 * 60 * 60)))
        # Rate limits (requests per window) — see api.py. Overridable via env.
        self.rate_limit_login = _get("RATE_LIMIT_LOGIN", "5/minute")
        self.rate_limit_runs = _get("RATE_LIMIT_RUNS", "20/minute")

        # --- LLM ---
        self.gemini_api_key = _get("GEMINI_API_KEY")
        self.gemini_model = _get("GEMINI_MODEL", "gemini-3.6-flash")

        # --- Adzuna (optional live-jobs feed) ---
        self.adzuna_app_id = _get("ADZUNA_APP_ID")
        self.adzuna_app_key = _get("ADZUNA_APP_KEY")
        self.adzuna_country = _get("ADZUNA_COUNTRY", "us")

    # --- derived / helpers ---
    @property
    def is_production(self):
        return self.env in ("prod", "production")

    def db_kwargs(self):
        """The kwargs for psycopg2.connect — the single definition of DB connection
        parameters, used by database.py."""
        return {
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
            "host": self.db_host,
            "port": self.db_port,
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