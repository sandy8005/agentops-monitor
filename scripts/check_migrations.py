"""
Migration path check — runs the REAL CLI (`python migrate.py`) against throwaway
databases and fails (exit 1) unless every path ends at the same schema:

  1. empty database        -> python migrate.py              (fresh install)
  2. already-current DB    -> python migrate.py              (must be a no-op)
  3. every older version V -> python migrate.py --to V --skip-checkpointer
                              (reproduces a deployment that stopped at V)
                           -> python migrate.py              (the upgrade)
                           -> python migrate.py              (no-op again)

After each path, `--status` must show nothing PENDING, and the resulting schema
(columns, constraints, indexes, triggers, functions) must be IDENTICAL to the fresh
install's, so an upgraded production database cannot drift from a new one.

    python scripts/check_migrations.py            # all older versions
    python scripts/check_migrations.py --quick    # fresh + re-run + previous version

Uses the DB_* settings to connect; the DB_USER needs CREATEDB. Creates and drops
databases named <DB_NAME>_migchk_<suffix>; never touches DB_NAME itself.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import psycopg2  # noqa: E402
from psycopg2 import sql  # noqa: E402

from migrate import _discover  # noqa: E402
from settings import settings  # noqa: E402

SCHEMA_QUERIES = {
    "columns": """
        SELECT table_name, column_name, data_type, is_nullable, column_default
        FROM information_schema.columns WHERE table_schema = 'public'
        ORDER BY 1, 2""",
    "constraints": """
        SELECT conrelid::regclass::text, conname, pg_get_constraintdef(oid)
        FROM pg_constraint WHERE connamespace = 'public'::regnamespace
        ORDER BY 1, 2""",
    "indexes": """
        SELECT tablename, indexname, indexdef FROM pg_indexes
        WHERE schemaname = 'public' ORDER BY 1, 2""",
    "triggers": """
        SELECT tgrelid::regclass::text, tgname, pg_get_triggerdef(oid)
        FROM pg_trigger WHERE NOT tgisinternal ORDER BY 1, 2""",
    "functions": """
        SELECT p.proname, pg_get_function_identity_arguments(p.oid), md5(p.prosrc)
        FROM pg_proc p WHERE p.pronamespace = 'public'::regnamespace
        ORDER BY 1, 2""",
}


def _admin():
    conn = psycopg2.connect(**settings.db_kwargs())
    conn.autocommit = True
    return conn


def _recreate(name):
    # Not `with conn:` — in psycopg2 >= 2.9 that opens a transaction block even in
    # autocommit mode, and CREATE/DROP DATABASE cannot run inside one.
    conn = _admin()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    finally:
        conn.close()


def _drop(name):
    conn = _admin()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))
    finally:
        conn.close()


def _cli(dbname, *args):
    env = dict(os.environ, DB_NAME=dbname)
    res = subprocess.run([sys.executable, os.path.join(ROOT, "migrate.py"), *args],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"`migrate.py {' '.join(args)}` on {dbname} exited "
                           f"{res.returncode}:\n{res.stdout}{res.stderr}")
    return res.stdout


def _schema(dbname):
    kw = dict(settings.db_kwargs(), dbname=dbname)
    conn = psycopg2.connect(**kw)
    try:
        out = {}
        with conn.cursor() as cur:
            for key, q in SCHEMA_QUERIES.items():
                cur.execute(q)
                out[key] = cur.fetchall()
        return out
    finally:
        conn.close()


def _assert_current(dbname, label):
    status = _cli(dbname, "--status")
    if "PENDING" in status:
        raise RuntimeError(f"{label}: migrations still pending:\n{status}")
    rerun = _cli(dbname)
    if "No pending migrations" not in rerun:
        raise RuntimeError(f"{label}: re-running migrate.py was not a no-op:\n{rerun}")


def _diff(a, b):
    lines = []
    for key in SCHEMA_QUERIES:
        only_a = sorted(set(a[key]) - set(b[key]))
        only_b = sorted(set(b[key]) - set(a[key]))
        for r in only_a[:10]:
            lines.append(f"  {key}: fresh only:    {r}")
        for r in only_b[:10]:
            lines.append(f"  {key}: upgraded only: {r}")
    return lines


def main(argv):
    settings.validate_db()
    quick = "--quick" in argv
    versions = [v for v, _n, _p in _discover()]
    older = versions[:-1][-1:] if quick else versions[:-1]
    base = settings.db_name
    failures = []

    fresh = f"{base}_migchk_fresh"
    created = [fresh]
    try:
        _recreate(fresh)
        _cli(fresh)
        _assert_current(fresh, "empty -> latest")
        reference = _schema(fresh)
        print(f"ok   empty database -> latest ({versions[-1]}), re-run is a no-op")

        for v in older:
            name = f"{base}_migchk_{v}"
            created.append(name)
            label = f"{v} -> latest"
            try:
                _recreate(name)
                _cli(name, "--to", v, "--skip-checkpointer")
                _cli(name)
                _assert_current(name, label)
                diff = _diff(reference, _schema(name))
                if diff:
                    failures.append(f"{label}: schema differs from a fresh install\n"
                                    + "\n".join(diff))
                    print(f"FAIL {label}: schema drift")
                else:
                    print(f"ok   {label}: identical to a fresh install")
            except RuntimeError as e:
                failures.append(f"{label}: {e}")
                print(f"FAIL {label}")
            finally:
                _drop(name)
    finally:
        for name in created:
            _drop(name)

    for f in failures:
        print("\nMIGRATION CHECK FAILED:", f)
    if not failures:
        print(f"migration check passed ({1 + len(older)} paths)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))