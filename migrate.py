"""
Migration runner.

Applies pending migrations from migrations/ in numeric order, tracking what's been
applied in the schema_migrations table. Idempotent: re-running only applies what's
new. Each migration file is named NNNN_name.py and defines `def upgrade(cur):`.

Usage:
    python migrate.py           # apply all pending migrations
    python migrate.py --status  # show applied vs pending, don't change anything

Fresh install:  run db_pg.py (builds the full schema and records 0001 as applied),
                then `python migrate.py` applies any 0002+ that exist.
Existing DB:    run `python migrate.py` — it applies 0001_baseline (idempotent, all
                IF NOT EXISTS) to bring the DB under management, then any 0002+.
"""
import os
import re
import sys
import importlib.util
from datetime import datetime

import psycopg2
from dotenv import load_dotenv

load_dotenv()

MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrations")
_FILE_RE = re.compile(r"^(\d{4})_([A-Za-z0-9_]+)\.py$")


def _connect():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"))


def _ensure_tracking_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            migration_name TEXT,
            applied_at TIMESTAMP DEFAULT NOW()
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
    conn = _connect()
    cur = conn.cursor()
    _ensure_tracking_table(cur)
    conn.commit()
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
                "VALUES (%s, %s, %s) ON CONFLICT (version) DO NOTHING",
                (version, name, datetime.now()),
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


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    else:
        migrate()