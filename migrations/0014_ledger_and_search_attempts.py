"""
Migration 0014 — stronger cost-ledger constraints and durable external-search
attempts. Idempotent.

Cost ledger (llm_calls / llm_cost_reservations / runs)
  * llm_calls.cost_usd NUMERIC(12,6) -> NUMERIC(18,10): six decimals rounded small
    calls DOWN (a $0.0000004 call was stored as $0) — the ledger under-counted.
  * Non-negative tokens, latency, cost and upper bounds.
  * cost_status semantics in the schema, not just in code:
        priced     -> cost_usd NOT NULL
        unknown    -> cost_usd NULL
        not_billed -> cost_usd NULL or 0 (nothing charged)
  * llm_calls.reservation_id REFERENCES llm_cost_reservations(id), UNIQUE: a
    reservation settles exactly one recorded call, never two.
  * llm_cost_reservations: amount_usd >= 0; status and settled_at agree
    (open <-> settled_at IS NULL).
  * runs.max_cost_usd >= 0; runs.cost_reserved_usd stays >= 0 (0012).
  * cost_ledger_violations(run_id): an auditable reconciliation query (the cached
    runs.cost_reserved_usd must equal the sum of OPEN reservations).

Existing rows are REPAIRED before the constraints are added, never silently
dropped, and always in the conservative direction (money is never lost from the
ledger, only moved into the "unknown, bounded" bucket):
  * unknown with a cost         -> cost moved to cost_upper_bound_usd, cost NULL
  * not_billed with a cost > 0  -> priced (a recorded charge means it was billed)
  * negative cost               -> cost NULL, unknown (an impossible value is not a price)
  * negative tokens / latency / upper bound / reservation amount / cap -> NULL
The number of repaired rows is printed.

external_search_attempts
  A provider request used to leave a durable trace only AFTER it returned
  (agent_searches). A worker that died mid-request left nothing, so the next
  generation could not tell "never called" from "called, outcome lost" and a crash
  loop could call a paid/quota-limited provider without bound. Every external
  request now writes a 'started' row BEFORE dispatch (fenced by execution
  generation) and closes it as succeeded / failed; a dead generation's 'started'
  rows become 'abandoned' (outcome unknown, counted against the per-search cap).
"""


def _repair(cur, label, sql):
    cur.execute(sql)
    if cur.rowcount:
        print(f"  repaired {cur.rowcount} {label}")


def upgrade(cur):
    # ---- repair rows that the new constraints would reject -------------------
    _repair(cur, "llm_calls row(s): negative cost -> unknown", """
        UPDATE llm_calls SET cost_usd = NULL, cost_status = 'unknown'
        WHERE cost_usd < 0""")
    _repair(cur, "llm_calls row(s): unknown cost had a value -> moved to upper bound", """
        UPDATE llm_calls SET cost_upper_bound_usd = GREATEST(COALESCE(cost_upper_bound_usd, 0),
                                                             cost_usd),
                             cost_usd = NULL
        WHERE cost_status = 'unknown' AND cost_usd IS NOT NULL""")
    _repair(cur, "llm_calls row(s): not_billed with a charge -> priced", """
        UPDATE llm_calls SET cost_status = 'priced'
        WHERE cost_status = 'not_billed' AND COALESCE(cost_usd, 0) <> 0""")
    _repair(cur, "llm_calls row(s): negative tokens/latency/bound -> NULL", """
        UPDATE llm_calls SET
            prompt_tokens = CASE WHEN prompt_tokens < 0 THEN NULL ELSE prompt_tokens END,
            completion_tokens = CASE WHEN completion_tokens < 0 THEN NULL
                                     ELSE completion_tokens END,
            latency_ms = CASE WHEN latency_ms < 0 THEN NULL ELSE latency_ms END,
            cost_upper_bound_usd = CASE WHEN cost_upper_bound_usd < 0 THEN NULL
                                        ELSE cost_upper_bound_usd END
        WHERE prompt_tokens < 0 OR completion_tokens < 0 OR latency_ms < 0
           OR cost_upper_bound_usd < 0""")
    _repair(cur, "llm_cost_reservations row(s): negative amount -> NULL", """
        UPDATE llm_cost_reservations SET amount_usd = NULL WHERE amount_usd < 0""")
    _repair(cur, "runs row(s): negative max_cost_usd -> NULL", """
        UPDATE runs SET max_cost_usd = NULL WHERE max_cost_usd < 0""")

    # ---- llm_calls -----------------------------------------------------------
    cur.execute("ALTER TABLE llm_calls ALTER COLUMN cost_usd TYPE NUMERIC(18,10)")
    cur.execute("ALTER TABLE llm_calls DROP CONSTRAINT IF EXISTS llm_calls_cost_status_chk")
    cur.execute("""
        ALTER TABLE llm_calls ADD CONSTRAINT llm_calls_cost_status_chk CHECK (
            cost_status IN ('priced', 'unknown', 'not_billed')
            AND (cost_status <> 'priced' OR cost_usd IS NOT NULL)
            AND (cost_status <> 'unknown' OR cost_usd IS NULL)
            AND (cost_status <> 'not_billed' OR COALESCE(cost_usd, 0) = 0)
        )""")
    cur.execute("ALTER TABLE llm_calls DROP CONSTRAINT IF EXISTS llm_calls_nonneg_chk")
    cur.execute("""
        ALTER TABLE llm_calls ADD CONSTRAINT llm_calls_nonneg_chk CHECK (
            COALESCE(cost_usd, 0) >= 0 AND COALESCE(cost_upper_bound_usd, 0) >= 0
            AND COALESCE(prompt_tokens, 0) >= 0 AND COALESCE(completion_tokens, 0) >= 0
            AND COALESCE(latency_ms, 0) >= 0
        )""")
    # A reservation can only be settled by one recorded call (FK + unique).
    cur.execute("UPDATE llm_calls c SET reservation_id = NULL WHERE reservation_id IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM llm_cost_reservations r WHERE r.id = c.reservation_id)")
    cur.execute("ALTER TABLE llm_calls DROP CONSTRAINT IF EXISTS llm_calls_reservation_fk")
    cur.execute("ALTER TABLE llm_calls ADD CONSTRAINT llm_calls_reservation_fk FOREIGN KEY "
                "(reservation_id) REFERENCES llm_cost_reservations(id) ON DELETE SET NULL")
    cur.execute("DROP INDEX IF EXISTS llm_calls_reservation_uniq")
    cur.execute("CREATE UNIQUE INDEX llm_calls_reservation_uniq ON llm_calls (reservation_id) "
                "WHERE reservation_id IS NOT NULL")

    # ---- llm_cost_reservations ----------------------------------------------
    cur.execute("UPDATE llm_cost_reservations SET settled_at = COALESCE(settled_at, created_at) "
                "WHERE status <> 'open'")
    cur.execute("UPDATE llm_cost_reservations SET settled_at = NULL WHERE status = 'open'")
    cur.execute("ALTER TABLE llm_cost_reservations DROP CONSTRAINT IF EXISTS "
                "llm_cost_reservations_ledger_chk")
    cur.execute("""
        ALTER TABLE llm_cost_reservations ADD CONSTRAINT llm_cost_reservations_ledger_chk CHECK (
            COALESCE(amount_usd, 0) >= 0
            AND execution_generation >= 0
            AND ((status = 'open') = (settled_at IS NULL))
        )""")

    # ---- runs ----------------------------------------------------------------
    cur.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_max_cost_chk")
    cur.execute("ALTER TABLE runs ADD CONSTRAINT runs_max_cost_chk CHECK "
                "(max_cost_usd IS NULL OR max_cost_usd >= 0)")
    cur.execute("""
        CREATE OR REPLACE FUNCTION cost_ledger_violations(p_run_id INTEGER)
        RETURNS TABLE (problem TEXT) LANGUAGE sql STABLE AS $$
            SELECT 'cost_reserved_usd ' || r.cost_reserved_usd || ' <> open reservations '
                   || COALESCE(o.total, 0)
            FROM runs r
            LEFT JOIN (SELECT run_id, SUM(amount_usd) AS total FROM llm_cost_reservations
                       WHERE status = 'open' GROUP BY run_id) o ON o.run_id = r.id
            WHERE r.id = p_run_id
              AND abs(r.cost_reserved_usd - COALESCE(o.total, 0)) > 1e-9
            UNION ALL
            SELECT 'settled reservation ' || res.id || ' has no recorded call'
            FROM llm_cost_reservations res
            WHERE res.run_id = p_run_id AND res.status = 'settled'
              AND NOT EXISTS (SELECT 1 FROM llm_calls c WHERE c.reservation_id = res.id)
            UNION ALL
            SELECT 'recorded call ' || c.id || ' references an OPEN reservation'
            FROM llm_calls c JOIN llm_cost_reservations res ON res.id = c.reservation_id
            WHERE c.run_id = p_run_id AND res.status = 'open'
        $$
    """)

    # ---- durable external-search attempts -----------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS external_search_attempts (
            id BIGSERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            iteration INTEGER NOT NULL,
            execution_generation INTEGER NOT NULL,
            provider TEXT NOT NULL CHECK (provider IN ('adzuna', 'remotive')),
            query_norm TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'started'
                CHECK (status IN ('started', 'succeeded', 'failed', 'abandoned')),
            provider_detail TEXT,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMPTZ,
            CONSTRAINT external_search_attempts_finish_chk
                CHECK ((status = 'started') = (finished_at IS NULL))
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS external_search_attempts_lookup_idx ON "
                "external_search_attempts (run_id, provider, query_norm)")
    # At most one in-flight request per (run, provider, query, generation).
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS external_search_attempts_inflight_uniq ON "
                "external_search_attempts (run_id, provider, query_norm, execution_generation) "
                "WHERE status = 'started'")