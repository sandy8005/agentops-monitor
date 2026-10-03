"""
Migration 0013 — finish the agent-only migration. Idempotent.

The controller agent is the only execution engine, but the database still said
otherwise: runs.mode defaulted to 'pipeline' (every helper-created run was a
legacy run until the API patched it), the 'pool' provider was still a valid goal
value, and legacy goal records were only "retired" lazily, if a worker happened to
pick them up.

  runs.mode DEFAULT 'agent'
        A run created by any helper is an agent run. 'pipeline' remains a value
        only for HISTORICAL rows, and only for retired ones (runs_mode_chk).

  runs.goal_retired_at / runs.goal_retired_reason
        Explicit retirement of legacy goal records:
          * every mode='pipeline' run                        -> 'pipeline_engine'
          * every goal naming a provider other than adzuna /
            remotive (the removed 'pool' practice source)    -> 'retired_provider'
        A retired run that had not finished is closed now (failed/engine_retired,
        or cancelled if the user had asked), its pending review is cleared and its
        open review cards are superseded — no worker ever has to discover it.

  runs_mode_chk: mode = 'agent' OR goal_retired_at IS NOT NULL
        No new non-agent run can be created.
"""

TERMINAL = ("success", "partial_success", "no_matches", "completed_with_errors",
            "cancelled", "failed")
LIVE_PROVIDERS = ("adzuna", "remotive")


def _in(values):
    return "(" + ", ".join("'" + v + "'" for v in values) + ")"


def upgrade(cur):
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS goal_retired_at TIMESTAMPTZ")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS goal_retired_reason TEXT")
    cur.execute("ALTER TABLE runs ALTER COLUMN mode SET DEFAULT 'agent'")

    # ---- explicit retirement of legacy goal records ------------------------
    cur.execute("""
        UPDATE runs SET goal_retired_at = NOW(), goal_retired_reason = 'pipeline_engine'
        WHERE mode <> 'agent' AND goal_retired_at IS NULL
    """)
    cur.execute(f"""
        UPDATE runs SET goal_retired_at = NOW(), goal_retired_reason = 'retired_provider'
        WHERE goal_retired_at IS NULL AND goal_json IS NOT NULL
          AND jsonb_typeof(goal_json->'providers') = 'array'
          AND EXISTS (SELECT 1 FROM jsonb_array_elements_text(goal_json->'providers') p
                      WHERE p NOT IN {_in(LIVE_PROVIDERS)})
    """)
    # Close every retired run that has not finished. A cancel the user asked for wins.
    cur.execute(f"""
        UPDATE runs SET
            status = CASE WHEN cancel_requested THEN 'cancelled' ELSE 'failed' END,
            error_code = CASE WHEN cancel_requested THEN 'cancelled' ELSE 'engine_retired' END,
            stop_reason = CASE WHEN cancel_requested THEN 'cancelled by user'
                ELSE 'engine_retired: this run''s goal uses a retired engine or job source ('
                     || goal_retired_reason || ') and was not executed; start a new run' END,
            pending_review = NULL,
            ended_at = COALESCE(ended_at, NOW())
        WHERE goal_retired_at IS NOT NULL AND status NOT IN {_in(TERMINAL)}
    """)
    # Open review cards of retired runs can never be answered: supersede them.
    cur.execute("""
        UPDATE review_requests SET status = 'superseded'
        WHERE status IN ('pending', 'submitted')
          AND run_id IN (SELECT id FROM runs WHERE goal_retired_at IS NOT NULL)
    """)

    cur.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_mode_chk")
    cur.execute("ALTER TABLE runs ADD CONSTRAINT runs_mode_chk CHECK "
                "(mode IN ('pipeline', 'agent') AND (mode = 'agent' OR goal_retired_at IS NOT NULL))")
    cur.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_goal_retired_chk")
    cur.execute("ALTER TABLE runs ADD CONSTRAINT runs_goal_retired_chk CHECK "
                "((goal_retired_at IS NULL) = (goal_retired_reason IS NULL) AND "
                "(goal_retired_reason IS NULL OR goal_retired_reason IN "
                "('pipeline_engine', 'retired_provider')))")