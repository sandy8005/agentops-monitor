"""
Migration runner.

Applies pending migrations from migrations/ in numeric order, tracking what's been
applied in the schema_migrations table. Idempotent: re-running only applies what's
new. Each migration file is named NNNN_name.py and defines `def upgrade(cur):`.

Usage:
    python migrate.py           # apply all pending migrations
    python migrate.py --status  # show applied vs pending, don't change anything
    python migrate.py --skip-checkpointer   # app migrations only

After the app migrations, LangGraph's own checkpoint tables are created/migrated
(PostgresSaver.setup(), via checkpointing.setup_schema). That DDL used to run on
EVERY graph start/resume; it is deployment work and now happens here (the worker
also runs it once at startup as a safety net).

Fresh install:  `python migrate.py` on an EMPTY database applies 0001_baseline and
                every later migration — this is the only supported way to build the
                schema (db_pg.py is just an alias for it).
Existing DB:    `python migrate.py` applies only what's pending. 0001_baseline is
                idempotent (IF NOT EXISTS) so a pre-migration DB is adopted safely.
"""
from contextlib import contextmanager
from timeutil import utcnow
import os
import re
import sys
import importlib.util
from datetime import datetime

import psycopg2

from settings import settings

MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrations")
_FILE_RE = re.compile(r"^(\d{4})_([A-Za-z0-9_]+)\.py$")


def _connect():
    settings.validate_db()
    return psycopg2.connect(**settings.db_kwargs())


# Advisory-lock key for ALL schema changes (app migrations + LangGraph checkpoint
# setup). Namespace 0x4147 ("AG") is shared with run_lock; key 0 is reserved for the
# schema and can never collide with a run id (run ids start at 1).
SCHEMA_LOCK_KEY = (0x4147, 0)
SCHEMA_LOCK_TIMEOUT_SECONDS = 600


@contextmanager
def schema_lock(timeout=SCHEMA_LOCK_TIMEOUT_SECONDS):
    """Hold the global schema-migration lock for the duration of the block.

    A SESSION-level advisory lock on a dedicated connection: it spans every
    per-migration transaction (discover -> read applied -> apply -> record), so two
    replicas deploying at the same time cannot both apply the same migration — the
    second one waits, then re-reads schema_migrations and finds nothing pending. If
    the holder dies, PostgreSQL releases the lock with its session."""
    conn = _connect()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout = %s", (f"{int(timeout)}s",))
            cur.execute("SELECT pg_advisory_lock(%s, %s)", SCHEMA_LOCK_KEY)
        try:
            yield
        finally:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s, %s)", SCHEMA_LOCK_KEY)
            except Exception:
                pass                         # closing the session releases it anyway
    finally:
        conn.close()


def _ensure_tracking_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            migration_name TEXT,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)


def _discover():
    """Return [(version, name, filepath)] sorted by version."""
    out = []
    for fn in os.listdir(MIGRATIONS_DIR):
        m = _FILE_RE.match(fn)
        if m:
            out.append((m.group(1), m.group(2), os.path.join(MIGRATIONS_DIR, fn)))
    return sorted(out, key=lambda t: t[0])


def _applied_versions(cur):
    cur.execute("SELECT version FROM schema_migrations")
    return {r[0] for r in cur.fetchall()}


def _load_upgrade(filepath):
    spec = importlib.util.spec_from_file_location("_mig", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "upgrade"):
        raise AttributeError(f"{filepath} has no upgrade(cur) function")
    return mod.upgrade


def status():
    conn = _connect()
    cur = conn.cursor()
    _ensure_tracking_table(cur)
    conn.commit()
    applied = _applied_versions(cur)
    conn.close()
    print("version  name                         status")
    print("-------  ---------------------------  -------")
    for version, name, _ in _discover():
        print(f"{version}     {name:<27}  {'applied' if version in applied else 'PENDING'}")


def migrate():
    """Apply every pending migration, holding the global schema lock throughout."""
    with schema_lock():
        _migrate_locked()


def _migrate_locked():
    conn = _connect()
    cur = conn.cursor()
    _ensure_tracking_table(cur)
    conn.commit()
    # Read the applied set only AFTER taking the lock: a concurrent deployer that
    # held it first has already applied (and recorded) what it found pending.
    applied = _applied_versions(cur)

    pending = [(v, n, p) for (v, n, p) in _discover() if v not in applied]
    if not pending:
        print("No pending migrations — database is up to date.")
        conn.close()
        return

    for version, name, filepath in pending:
        print(f"Applying {version}_{name} ...")
        upgrade = _load_upgrade(filepath)
        try:
            upgrade(cur)                     # each migration is one transaction
            cur.execute(
                "INSERT INTO schema_migrations (version, migration_name, applied_at) "
                "VALUES (%s, %s, %s)",
                (version, name, utcnow()),
            )
            conn.commit()
            print(f"  applied {version}_{name}")
        except Exception as e:
            conn.rollback()
            print(f"  FAILED {version}_{name} — rolled back: {e}")
            conn.close()
            sys.exit(1)

    conn.close()
    print("All migrations applied.")


def setup_checkpointer():
    """Create/upgrade LangGraph's checkpoint tables (idempotent)."""
    try:
        from checkpointing import setup_schema
        setup_schema()
        print("LangGraph checkpoint schema is up to date.")
    except Exception as e:
        print(f"  FAILED LangGraph checkpoint setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    else:
        migrate()
        if "--skip-checkpointer" not in sys.argv:
            setup_checkpointer()