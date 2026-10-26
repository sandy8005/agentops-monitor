"""
Migration 0019 — a status is never NULL. Idempotent.

`status <> 'success'` is NULL (not true) when status is NULL, so a row with no
status dropped out of every failure count (smoke_live's checks, the retired-run
backstop in worker.py). The four status columns that still allowed NULL get an
explicit value and NOT NULL:

  llm_calls.status    NULL -> 'unknown'; default 'success' -> 'unknown' (a writer
                      that forgets the status must not record a success). Legacy
                      NULL rows also lose their fabricated 0 tokens / $0, exactly as
                      0012 did for failed rows (cost_status is never 'priced' there).
  tool_calls.status   NULL -> 'unknown'; default 'unknown'.
  steps.status        NULL -> 'unknown'; default 'unknown'.
  runs.status         NULL -> 'failed' / internal_error, ended (a run with no state
                      cannot be resumed); no default: create_run_tx always says
                      'queued'. runs_status_chk already lists the legal values.

'unknown' is deliberately not 'success': every failure-detection query treats it
as not successful. The number of repaired rows is printed.

cost_ledger_violations() needs no change: its live definition (0014) compares only
NOT NULL columns (reservation status, cost_reserved_usd). The `status <> 'success'`
in 0012 was a one-time backfill, covered for NULL rows by the repair below.
"""


def _repair(cur, label, sql):
    cur.execute(sql)
    if cur.rowcount:
        print(f"  repaired {cur.rowcount} {label}")


def upgrade(cur):
    _repair(cur, "llm_calls row(s): NULL status had fabricated 0 tokens -> NULL", """
        UPDATE llm_calls SET prompt_tokens = NULL, completion_tokens = NULL, cost_usd = NULL
        WHERE status IS NULL AND cost_status <> 'priced'
          AND COALESCE(cost_usd, 0) = 0 AND COALESCE(prompt_tokens, 0) = 0
          AND COALESCE(completion_tokens, 0) = 0
          AND (prompt_tokens IS NOT NULL OR completion_tokens IS NOT NULL
               OR cost_usd IS NOT NULL)""")
    for table in ("llm_calls", "tool_calls", "steps"):
        _repair(cur, f"{table} row(s): NULL status -> unknown",
                f"UPDATE {table} SET status = 'unknown' WHERE status IS NULL")
        cur.execute(f"ALTER TABLE {table} ALTER COLUMN status SET DEFAULT 'unknown'")
        cur.execute(f"ALTER TABLE {table} ALTER COLUMN status SET NOT NULL")

    _repair(cur, "runs row(s): NULL status -> failed", """
        UPDATE runs SET status = 'failed',
                        error_code = COALESCE(error_code, 'internal_error'),
                        ended_at = COALESCE(ended_at, NOW())
        WHERE status IS NULL""")
    cur.execute("ALTER TABLE runs ALTER COLUMN status SET NOT NULL")
