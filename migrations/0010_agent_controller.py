"""
Migration 0010 — controller loop, bound reviews (R01), durable budget (R16),
execution generations (R28), structured resume suggestions. Idempotent.
"""
RUN_STATUSES = ("queued", "running", "retrying", "waiting_for_human", "success",
                "partial_success", "completed_with_errors", "failed", "cancelled",
                "no_matches")


def _in(values):
    return "(" + ", ".join("'" + v + "'" for v in values) + ")"


def upgrade(cur):
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'pipeline'")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS goal_json JSONB")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS llm_call_budget INTEGER")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS llm_calls_reserved INTEGER NOT NULL DEFAULT 0")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS execution_generation INTEGER NOT NULL DEFAULT 0")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS controller_mode TEXT")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS goal_progress JSONB")
    cur.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_status_chk")
    cur.execute(f"ALTER TABLE runs ADD CONSTRAINT runs_status_chk CHECK (status IN {_in(RUN_STATUSES)})")
    cur.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_mode_chk")
    cur.execute("ALTER TABLE runs ADD CONSTRAINT runs_mode_chk CHECK (mode IN ('pipeline', 'agent'))")
    cur.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_budget_chk")
    cur.execute("ALTER TABLE runs ADD CONSTRAINT runs_budget_chk CHECK "
                "(llm_calls_reserved >= 0 AND (llm_call_budget IS NULL OR llm_call_budget >= 0))")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_actions (
            id SERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            iteration INTEGER NOT NULL,
            execution_generation INTEGER NOT NULL,
            action TEXT NOT NULL,
            arguments JSONB,
            reason TEXT,
            decided_by TEXT NOT NULL,
            status TEXT NOT NULL,
            observation JSONB,
            error TEXT,
            step_id INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMPTZ,
            CONSTRAINT agent_actions_iter_uniq UNIQUE (run_id, iteration),
            CONSTRAINT agent_actions_status_chk
                CHECK (status IN ('proposed', 'rejected', 'executed', 'failed')),
            CONSTRAINT agent_actions_decider_chk
                CHECK (decided_by IN ('llm', 'rules', 'backend'))
        )""")
    cur.execute("CREATE INDEX IF NOT EXISTS agent_actions_run_idx ON agent_actions (run_id, iteration)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_searches (
            id SERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            query_norm TEXT NOT NULL,
            location TEXT,
            iteration INTEGER NOT NULL,
            provider_status TEXT NOT NULL,
            new_jobs INTEGER NOT NULL DEFAULT 0,
            duplicates INTEGER NOT NULL DEFAULT 0,
            eligible_jobs INTEGER NOT NULL DEFAULT 0,
            rejection_summary JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT agent_searches_uniq UNIQUE (run_id, provider, query_norm)
        )""")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS review_requests (
            review_id TEXT PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            step_id INTEGER,
            job_id INTEGER,
            payload JSONB,
            status TEXT NOT NULL DEFAULT 'pending',
            decision TEXT,
            answer TEXT,
            comment TEXT,
            reviewer_user_id INTEGER,
            reviewer TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            submitted_at TIMESTAMPTZ,
            consumed_at TIMESTAMPTZ,
            CONSTRAINT review_requests_kind_chk CHECK (kind IN ('job_review', 'input_request')),
            CONSTRAINT review_requests_status_chk
                CHECK (status IN ('pending', 'submitted', 'consumed', 'superseded')),
            CONSTRAINT review_requests_decision_chk
                CHECK (decision IS NULL OR decision IN ('Apply', 'Maybe', 'Skip'))
        )""")
    cur.execute("CREATE INDEX IF NOT EXISTS review_requests_run_idx ON review_requests (run_id, status)")
    cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS review_requests_one_open_per_run
                   ON review_requests (run_id) WHERE status IN ('pending', 'submitted')""")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS resume_suggestions (
            id SERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            job_id INTEGER NOT NULL,
            resume_id INTEGER NOT NULL,
            resume_hash TEXT NOT NULL,
            position INTEGER NOT NULL,
            kind TEXT NOT NULL,
            original_text TEXT,
            suggested_text TEXT NOT NULL,
            reason TEXT NOT NULL,
            evidence JSONB NOT NULL,
            method TEXT NOT NULL,
            status TEXT NOT NULL,
            validation_notes TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT resume_suggestions_uniq UNIQUE (run_id, job_id, position),
            CONSTRAINT resume_suggestions_method_chk CHECK (method IN ('rules', 'gemini')),
            CONSTRAINT resume_suggestions_status_chk
                CHECK (status IN ('validated', 'needs_confirmation', 'rejected'))
        )""")
    cur.execute("CREATE INDEX IF NOT EXISTS resume_suggestions_run_idx ON resume_suggestions (run_id, job_id)")