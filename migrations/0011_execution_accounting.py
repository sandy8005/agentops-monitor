"""
Migration 0011 — execution accounting for the autonomous controller. Idempotent.

  runs.active_runtime_seconds / runs.execution_started_at
        The runtime limit measured wall-clock time since the run FIRST started, so
        a run that worked 3 minutes and then waited an hour for a human was treated
        as having used 63 minutes. Runtime is now accumulated per execution
        interval: begin_execution() opens an interval, mark_waiting()/finalize()
        close it. Human-review time is never charged.

  runs.unknown_cost_calls
        SUM(cost_usd) ignores NULLs, so a run whose calls were partly unpriced
        looked like a complete total. The count of successful-but-unpriced calls is
        stored next to total_cost so every reader can tell partial from complete.

  agent_action_attempts
        agent_actions is the controller DECISION for (run, iteration) and is reused
        on replay. Each EXECUTION of that decision (one per execution generation) is
        now its own row, so "attempt 1 failed, attempt 2 succeeded" is durable.

  agent_searches: one row per (run, provider, query, execution_generation)
        A search that failed transiently in one worker generation used to block the
        same search forever (UNIQUE(run, provider, query) + "never refetch on
        replay"). Rows are now generation-scoped and record whether the provider was
        actually called (fetched) or a previous result was reused.
"""


def upgrade(cur):
    # ---- runtime accounting --------------------------------------------------
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS active_runtime_seconds "
                "DOUBLE PRECISION NOT NULL DEFAULT 0")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS execution_started_at TIMESTAMPTZ")
    # A run executing during the upgrade gets an open interval from now on (we cannot
    # know how much of its past wall-clock time was active).
    cur.execute("""UPDATE runs SET execution_started_at = NOW()
                   WHERE status = 'running' AND execution_started_at IS NULL""")
    cur.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_active_runtime_chk")
    cur.execute("ALTER TABLE runs ADD CONSTRAINT runs_active_runtime_chk "
                "CHECK (active_runtime_seconds >= 0)")

    # ---- cost completeness ---------------------------------------------------
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS unknown_cost_calls "
                "INTEGER NOT NULL DEFAULT 0")
    cur.execute("""
        UPDATE runs r SET unknown_cost_calls = c.n
        FROM (SELECT run_id, COUNT(*) AS n FROM llm_calls
              WHERE cost_usd IS NULL AND status = 'success' GROUP BY run_id) c
        WHERE c.run_id = r.id
    """)

    # ---- action decisions vs. execution attempts -----------------------------
    cur.execute("ALTER TABLE agent_actions ADD COLUMN IF NOT EXISTS attempt_count "
                "INTEGER NOT NULL DEFAULT 0")
    cur.execute("ALTER TABLE agent_actions ADD COLUMN IF NOT EXISTS last_execution_generation INTEGER")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_action_attempts (
            id SERIAL PRIMARY KEY,
            action_id INTEGER NOT NULL REFERENCES agent_actions(id) ON DELETE CASCADE,
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            iteration INTEGER NOT NULL,
            execution_generation INTEGER NOT NULL,
            run_attempt INTEGER,
            attempt_number INTEGER NOT NULL,
            decision_replayed BOOLEAN NOT NULL DEFAULT FALSE,
            status TEXT NOT NULL,
            observation JSONB,
            error TEXT,
            step_id INTEGER,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMPTZ,
            CONSTRAINT agent_action_attempts_gen_uniq UNIQUE (action_id, execution_generation),
            CONSTRAINT agent_action_attempts_status_chk
                CHECK (status IN ('running', 'rejected', 'executed', 'failed'))
        )""")
    cur.execute("CREATE INDEX IF NOT EXISTS agent_action_attempts_run_idx "
                "ON agent_action_attempts (run_id, iteration, attempt_number)")
    # Backfill: every finished legacy action becomes attempt #1 of its decision.
    cur.execute("""
        INSERT INTO agent_action_attempts (action_id, run_id, iteration, execution_generation,
            attempt_number, status, observation, error, step_id, started_at, finished_at)
        SELECT a.id, a.run_id, a.iteration, a.execution_generation, 1, a.status,
               a.observation, a.error, a.step_id, a.created_at, a.finished_at
        FROM agent_actions a
        WHERE a.status IN ('rejected', 'executed', 'failed')
          AND NOT EXISTS (SELECT 1 FROM agent_action_attempts t WHERE t.action_id = a.id)
    """)
    cur.execute("""UPDATE agent_actions SET attempt_count = 1,
                          last_execution_generation = execution_generation
                   WHERE status IN ('rejected', 'executed', 'failed') AND attempt_count = 0""")

    # ---- generation-scoped searches ------------------------------------------
    cur.execute("ALTER TABLE agent_searches ADD COLUMN IF NOT EXISTS execution_generation "
                "INTEGER NOT NULL DEFAULT 0")
    cur.execute("ALTER TABLE agent_searches ADD COLUMN IF NOT EXISTS provider_detail TEXT")
    cur.execute("ALTER TABLE agent_searches ADD COLUMN IF NOT EXISTS fetched "
                "BOOLEAN NOT NULL DEFAULT TRUE")
    cur.execute("ALTER TABLE agent_searches ADD COLUMN IF NOT EXISTS reused_from_generation INTEGER")
    cur.execute("ALTER TABLE agent_searches DROP CONSTRAINT IF EXISTS agent_searches_uniq")
    cur.execute("ALTER TABLE agent_searches DROP CONSTRAINT IF EXISTS agent_searches_gen_uniq")
    cur.execute("ALTER TABLE agent_searches ADD CONSTRAINT agent_searches_gen_uniq "
                "UNIQUE (run_id, provider, query_norm, execution_generation)")
    cur.execute("CREATE INDEX IF NOT EXISTS agent_searches_lookup_idx "
                "ON agent_searches (run_id, provider, query_norm, execution_generation DESC)")