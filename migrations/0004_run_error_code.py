"""
Migration 0004 — runs.error_code.

error_codes.py introduced a machine-readable classification of WHY a run ended in
a non-success state, and llm.finish_run() now writes it:

    UPDATE runs SET error_code = %s WHERE id = %s

but the column was never added to the schema. Until this migration runs, that
UPDATE fails with `column "error_code" does not exist`, so any finish_run() that
passes an error_code (LLM unavailable, quota exhausted, cancelled, ...) errors out.

error_code is the stable, queryable vocabulary (distinct from the coarse `status`
and the human-readable `stop_reason`) that you can group/alert on:

    SELECT error_code, count(*) FROM runs WHERE error_code IS NOT NULL GROUP BY 1;

Values are the plain strings from error_codes.ErrorCode (e.g. 'llm_unavailable'),
so TEXT is the right type; NULL means "no error" (a successful run).

Idempotent: ADD COLUMN IF NOT EXISTS. Applies to both fresh installs (after
db_pg.py records 0001) and existing databases.
"""


def upgrade(cur):
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS error_code TEXT")