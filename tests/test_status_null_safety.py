"""Round 6, backlog E: a NULL status is never read as success and never drops out
of a failure count.

`status <> 'success'` is NULL (not true) for a NULL status, so such rows vanished
from every failure-detection query. Migration 0019 makes the trace/run status
columns NOT NULL (backfilling legacy NULLs to an explicit non-success value) and the
failure-detection queries use IS DISTINCT FROM, so the guarantee holds twice: in
the schema and in the query."""
import importlib.util
import os
import re
import uuid

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NULLABLE_BEFORE_0019 = [("llm_calls", "status"), ("tool_calls", "status"),
                        ("steps", "status"), ("runs", "status")]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- static guard (no database) ---------------------------------------------

# Status inequalities that are safe because the column is NOT NULL with a CHECK
# (a NULL can never be stored), documented next to each query.
_ALLOWED = {
    ("api.py", "status <> 'rejected'"),        # resume_suggestions.status NOT NULL + CHECK
}
# SQL only: an upper-case NOT IN (Python's `not in` on a variable is not SQL).
_INEQ = re.compile(r"\bstatus\s*(?:<>|!=)\s*'[a-z_]+'|\bstatus\s+NOT\s+IN\b")


def _runtime_sources():
    for base, dirs, files in os.walk(ROOT):
        rel = os.path.relpath(base, ROOT)
        top = rel.split(os.sep)[0]
        if rel != "." and (top in ("venv", "tests", "migrations") or top.startswith(".")):
            dirs[:] = []
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.relpath(os.path.join(base, f), ROOT).replace(os.sep, "/")


def test_no_null_unsafe_status_inequality_in_runtime_sql():
    found = []
    for rel in _runtime_sources():
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            for m in _INEQ.finditer(fh.read()):
                if (rel, m.group(0)) not in _ALLOWED:
                    found.append((rel, m.group(0)))
    assert found == [], found


# --- schema -----------------------------------------------------------------

def _cur():
    from database import get_connection
    conn = get_connection()
    return conn, conn.cursor()


@pytest.mark.db
@pytest.mark.parametrize("table,column", NULLABLE_BEFORE_0019)
def test_status_columns_reject_null(table, column):
    conn, cur = _cur()
    try:
        cur.execute("SELECT is_nullable FROM information_schema.columns WHERE "
                    "table_schema = 'public' AND table_name = %s AND column_name = %s",
                    (table, column))
        assert cur.fetchone() == ("NO",)
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.db
def test_unstated_llm_call_status_is_unknown_not_success():
    """The old column default was 'success': a writer that forgot the status
    recorded a success. The fail-closed default is 'unknown'."""
    from auth import create_user
    uid = create_user("nul_" + uuid.uuid4().hex[:10], "password-1234")
    conn, cur = _cur()
    try:
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                    "VALUES ('running', 't', %s, 'agent') RETURNING id", (uid,))
        rid = cur.fetchone()[0]
        cur.execute("INSERT INTO llm_calls (run_id, cost_status) VALUES (%s, 'unknown') "
                    "RETURNING status", (rid,))
        assert cur.fetchone()[0] == "unknown"
        cur.execute("INSERT INTO tool_calls (run_id, tool_name) VALUES (%s, 't') "
                    "RETURNING status", (rid,))
        assert cur.fetchone()[0] == "unknown"
        cur.execute("INSERT INTO steps (run_id, step_name, step_order) VALUES (%s, 's', 0) "
                    "RETURNING status", (rid,))
        assert cur.fetchone()[0] == "unknown"
    finally:
        conn.rollback()
        conn.close()


def _legacy_null_rows(cur):
    """Inside the caller's transaction: re-open the pre-0019 schema (nullable
    status, 'success' default) and insert one NULL-status row per table. Rolled
    back by the caller, so the shared schema is never changed."""
    from auth import create_user
    uid = create_user("nul_" + uuid.uuid4().hex[:10], "password-1234")
    for table, column in NULLABLE_BEFORE_0019:
        cur.execute(f"ALTER TABLE {table} ALTER COLUMN {column} DROP NOT NULL")
    cur.execute("ALTER TABLE llm_calls ALTER COLUMN status SET DEFAULT 'success'")
    cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                "VALUES ('running', 't', %s, 'agent') RETURNING id", (uid,))
    rid = cur.fetchone()[0]
    cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                "VALUES (NULL, 't', %s, 'agent') RETURNING id", (uid,))
    null_run = cur.fetchone()[0]
    cur.execute("INSERT INTO llm_calls (run_id, status, cost_status, prompt_tokens, "
                "completion_tokens, logical_call_id) "
                "VALUES (%s, NULL, 'unknown', 0, 0, 'lc-1') RETURNING id", (rid,))
    call = cur.fetchone()[0]
    cur.execute("INSERT INTO tool_calls (run_id, tool_name, input_json, status) "
                "VALUES (%s, 't', '{}', NULL) RETURNING id", (rid,))
    tool = cur.fetchone()[0]
    cur.execute("INSERT INTO steps (run_id, step_name, step_order, status) "
                "VALUES (%s, 's', 0, NULL) RETURNING id", (rid,))
    step = cur.fetchone()[0]
    return rid, null_run, call, tool, step


@pytest.mark.db
def test_smoke_queries_count_null_status_as_failure():
    smoke = _load(("scripts", "smoke_live.py"), "smoke_live_nulls")
    conn, cur = _cur()
    try:
        rid, *_ = _legacy_null_rows(cur)
        counts = {}
        for name in ("llm_failed_unrecovered", "tool_failed_unrecovered",
                     "failed_steps_unrecovered"):
            cur.execute(smoke.RUN_QUERIES[name], {"r": rid})
            counts[name] = cur.fetchone()[0]
        assert counts == {"llm_failed_unrecovered": 1, "tool_failed_unrecovered": 1,
                          "failed_steps_unrecovered": 1}
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.db
def test_migration_0019_backfills_null_statuses_idempotently():
    mig = _load(("migrations", "0019_status_not_null.py"), "mig_0019_under_test")
    conn, cur = _cur()
    try:
        rid, null_run, call, tool, step = _legacy_null_rows(cur)
        mig.upgrade(cur)
        mig.upgrade(cur)                                   # re-run is a no-op
        cur.execute("SELECT status, prompt_tokens, completion_tokens FROM llm_calls "
                    "WHERE id = %s", (call,))
        assert cur.fetchone() == ("unknown", None, None)   # fabricated zeros cleared
        cur.execute("SELECT status FROM tool_calls WHERE id = %s", (tool,))
        assert cur.fetchone() == ("unknown",)
        cur.execute("SELECT status FROM steps WHERE id = %s", (step,))
        assert cur.fetchone() == ("unknown",)
        cur.execute("SELECT status, error_code, ended_at IS NOT NULL FROM runs "
                    "WHERE id = %s", (null_run,))
        assert cur.fetchone() == ("failed", "internal_error", True)
        cur.execute("SELECT status FROM runs WHERE id = %s", (rid,))
        assert cur.fetchone() == ("running",)              # untouched
        cur.execute("SELECT column_default FROM information_schema.columns WHERE "
                    "table_name = 'llm_calls' AND column_name = 'status'")
        assert "unknown" in cur.fetchone()[0]
    finally:
        conn.rollback()
        conn.close()
