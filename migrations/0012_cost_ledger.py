"""
Migration 0012 — enforceable USD limits, honest cost states, single engine. Idempotent.

  llm_calls.cost_status  ('priced' | 'unknown' | 'not_billed')
  llm_calls.cost_upper_bound_usd
  llm_calls.usage_missing, llm_calls.execution_generation, llm_calls.reservation_id
        A call whose provider usage was missing, or that failed AFTER dispatch
        (timeout, connection reset, 5xx), used to be stored as 0 tokens / $0 and
        was invisible to the fail-closed cost check. Such calls are now
        cost_status='unknown' with NULL tokens/cost, plus the conservative upper
        bound that was reserved for them. Requests the provider provably rejected
        before doing work (429 / 401 / 403 / 400) are 'not_billed'.

  llm_cost_reservations  + runs.cost_reserved_usd / runs.max_cost_usd
        The USD cap was checked against spend ALREADY recorded, so the request
        that crossed the cap was always allowed. Every request now atomically
        reserves its maximum possible cost first (spent + reserved + projected
        <= cap, under the run row lock, fenced by execution generation) and the
        reservation is settled in the same transaction that records the call.

  runs.execution_heartbeat_at
        A dead worker's open runtime interval is charged up to its last heartbeat
        plus a grace period, not until a replacement worker happens to start.

  job_postings.source  — the 'seed' default is dropped: only live providers
        (adzuna, remotive) insert postings now.

Backfill of legacy llm_calls rows: success + price -> priced; success without a
price -> unknown; failures whose message shows the provider rejected the request
(429/401/403/400) -> not_billed; any other legacy failure -> unknown (the old
code recorded it as $0, which we can no longer trust).
"""


def upgrade(cur):
    # ---- llm_calls: explicit cost state ------------------------------------
    cur.execute("ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS cost_status TEXT")
    cur.execute("ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS cost_upper_bound_usd "
                "DOUBLE PRECISION")
    cur.execute("ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS usage_missing BOOLEAN "
                "NOT NULL DEFAULT FALSE")
    cur.execute("ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS execution_generation INTEGER")
    cur.execute("ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS reservation_id BIGINT")
    cur.execute("""
        UPDATE llm_calls SET cost_status = CASE
            WHEN status = 'success' AND cost_usd IS NOT NULL THEN 'priced'
            WHEN status = 'success' THEN 'unknown'
            WHEN error_message ~* '(\\m429\\M|RESOURCE_EXHAUSTED|\\m401\\M|\\m403\\M|\\m400\\M|INVALID_ARGUMENT|PERMISSION_DENIED|UNAUTHENTICATED)'
                THEN 'not_billed'
            ELSE 'unknown' END
        WHERE cost_status IS NULL
    """)
    # Legacy failed rows carried a fabricated 0 tokens / $0 — clear them.
    cur.execute("""UPDATE llm_calls SET prompt_tokens = NULL, completion_tokens = NULL,
                          cost_usd = NULL
                   WHERE status <> 'success' AND COALESCE(cost_usd, 0) = 0
                     AND COALESCE(prompt_tokens, 0) = 0""")
    cur.execute("ALTER TABLE llm_calls ALTER COLUMN cost_status SET NOT NULL")
    # A writer that does not state the cost status gets the fail-closed value.
    cur.execute("ALTER TABLE llm_calls ALTER COLUMN cost_status SET DEFAULT 'unknown'")
    cur.execute("ALTER TABLE llm_calls DROP CONSTRAINT IF EXISTS llm_calls_cost_status_chk")
    cur.execute("""ALTER TABLE llm_calls ADD CONSTRAINT llm_calls_cost_status_chk
                   CHECK (cost_status IN ('priced', 'unknown', 'not_billed')
                          AND (cost_status <> 'priced' OR cost_usd IS NOT NULL))""")
    cur.execute("CREATE INDEX IF NOT EXISTS llm_calls_run_cost_idx "
                "ON llm_calls (run_id, cost_status)")

    # ---- runs: reservation ledger totals / cap / heartbeat -----------------
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS cost_reserved_usd "
                "DOUBLE PRECISION NOT NULL DEFAULT 0")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS max_cost_usd DOUBLE PRECISION")
    cur.execute("""UPDATE runs SET max_cost_usd = (goal_json->'limits'->>'max_cost_usd')::float
                   WHERE max_cost_usd IS NULL AND goal_json IS NOT NULL
                     AND goal_json->'limits' ? 'max_cost_usd'""")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS execution_heartbeat_at TIMESTAMPTZ")
    cur.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_cost_reserved_chk")
    cur.execute("ALTER TABLE runs ADD CONSTRAINT runs_cost_reserved_chk "
                "CHECK (cost_reserved_usd >= 0)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_cost_reservations (
            id BIGSERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            execution_generation INTEGER NOT NULL,
            operation TEXT,
            -- Maximum possible cost of the request. NULL only when no USD cap is set
            -- and the model's price is unknown (nothing to enforce).
            amount_usd DOUBLE PRECISION,
            -- open      : request may be in flight — counted against the cap
            -- settled   : the call was recorded in llm_calls (by reservation_id)
            -- abandoned : the worker died/lost the run before recording the call;
            --             counted against the cap forever (the request may have
            --             been billed) and reported as unknown cost
            status TEXT NOT NULL DEFAULT 'open'
                CHECK (status IN ('open', 'settled', 'abandoned')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            settled_at TIMESTAMPTZ
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS llm_cost_reservations_run_idx "
                "ON llm_cost_reservations (run_id, status)")

    # ---- job_postings: no more practice/seed defaults ----------------------
    cur.execute("ALTER TABLE job_postings ALTER COLUMN source DROP DEFAULT")