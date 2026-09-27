"""
Migration 0008 — integrity, observability and time-zone hardening.

Adds the columns the review fixes need, turns application-only invariants into
database constraints, indexes the Monitor's hot query paths, and moves every
timestamp to TIMESTAMPTZ. Idempotent: safe to re-run.

  Columns
    steps.security_flag / security_reason   security WARNING (continue + record),
                                            kept distinct from needs_human_review
                                            (pause + human approval).
    steps.run_attempt, llm_calls.run_attempt, tool_calls.run_attempt
                                            which execution attempt of a run wrote
                                            the trace row, so retries don't blur the
                                            timeline.
    steps.reviewer_user_id                  WHO made a human decision (not "human").
    runs.attempt                            the run's current execution attempt.
    runs.last_attempt_ended_at              when the previous attempt ended; ended_at
                                            is reserved for the run's FINAL end.
    job_reqs_cache.expires_at               rule-based fallback rows expire; LLM rows
                                            don't (NULL = never).
    job_searches.location_filter_applied    FALSE for providers (Remotive) that don't
                                            geo-filter, so the association isn't read
                                            as "returned for this location".

  Constraints (added NOT VALID, then validated when existing data allows — a
  violation in legacy rows leaves the constraint enforcing NEW writes and is
  reported, instead of failing the whole migration).

  Indexes for steps/llm_calls/tool_calls/evaluations/job_queue access paths —
  PostgreSQL does NOT index referencing FK columns automatically.

  TIMESTAMPTZ: existing naive values were written with datetime.now(), i.e. in the
  APPLICATION SERVER's local zone. They are interpreted in LEGACY_TIMESTAMP_TZ
  (falling back to the DB session TimeZone, with a warning), then stored as
  absolute time.

  Uniqueness constraints are REQUIRED: duplicate legacy rows are remediated and
  the constraint added, or the migration fails and is not recorded as applied.
"""

# Tables owned by this app (LangGraph's checkpoint tables are left alone).
_APP_TABLES = (
    "runs", "steps", "llm_calls", "tool_calls", "job_searches", "job_search_results",
    "job_postings", "resumes", "evaluations", "parsed_resume_cache", "job_reqs_cache",
    "users", "run_rankings", "run_advice", "schema_migrations", "job_queue",
)

RUN_STATUSES = ("queued", "running", "retrying", "waiting_for_human", "success",
                "completed_with_errors", "failed", "cancelled", "no_matches")
QUEUE_STATUSES = ("queued", "running", "done", "failed", "cancelled")
DECISIONS = ("Apply", "Maybe", "Skip")


def _in(values):
    return "(" + ", ".join("'" + v + "'" for v in values) + ")"


def _constraint_exists(cur, name):
    cur.execute("SELECT 1 FROM pg_constraint WHERE conname = %s", (name,))
    return cur.fetchone() is not None


def _add_constraint(cur, table, name, definition):
    """Add a constraint NOT VALID (enforced for new rows immediately), then try to
    VALIDATE it against existing rows inside a savepoint."""
    if not _constraint_exists(cur, name):
        cur.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} {definition} NOT VALID")
    cur.execute("SAVEPOINT v")
    try:
        cur.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}")
        cur.execute("RELEASE SAVEPOINT v")
    except Exception as e:  # legacy rows violate it — keep it enforcing new writes
        cur.execute("ROLLBACK TO SAVEPOINT v")
        print(f"  WARNING: {name} added but NOT validated (existing rows violate it): {e}")


# How to remediate duplicate legacy rows before a UNIQUE constraint is added:
# (table, constraint) -> "keep the row with the lowest/highest id in each group".
# run_rankings keeps its FIRST row per key (the original ranking position); for
# run_advice the NEWEST row is the one the app would have shown.
_DEDUP_KEEP = {
    "run_rankings_run_pos_uniq": "min",
    "run_rankings_run_job_uniq": "min",
    "run_advice_run_job_uniq": "max",
}


def _remove_duplicates(cur, table, name, cols):
    """Delete duplicate rows for `cols`, keeping one per group (see _DEDUP_KEEP).
    Rows whose key contains NULL are never duplicates for a UNIQUE constraint and are
    left alone. Returns the number of rows removed."""
    keep = _DEDUP_KEEP.get(name, "min")
    col_list = [c.strip() for c in cols.split(",")]
    not_null = " AND ".join(f"{c} IS NOT NULL" for c in col_list)
    cur.execute(f"""
        DELETE FROM {table} t
        USING (
            SELECT {cols}, {keep}(id) AS keep_id
            FROM {table} WHERE {not_null}
            GROUP BY {cols} HAVING count(*) > 1
        ) d
        WHERE {" AND ".join(f"t.{c} = d.{c}" for c in col_list)}
          AND t.id <> d.keep_id
    """)
    return cur.rowcount


def _add_unique(cur, table, name, cols):
    """
    Add a UNIQUE constraint — and GUARANTEE it exists when this returns.

    Uniqueness is an invariant, so it is never silently skipped: duplicate legacy
    rows are REMEDIATED first (deterministically, see _DEDUP_KEEP), then the
    constraint is added. If it still can't be added, the exception propagates and
    migrate.py rolls the whole migration back WITHOUT recording it as applied — so
    two databases that both report "0008 applied" really do have the same schema.
    """
    if _constraint_exists(cur, name):
        return
    removed = _remove_duplicates(cur, table, name, cols)
    if removed:
        print(f"  {name}: removed {removed} duplicate legacy row(s) from {table}")
    cur.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} UNIQUE ({cols})")


def _legacy_timezone(cur):
    """
    The time zone the legacy NAIVE timestamps were WRITTEN in.

    Old code wrote datetime.now() — the APPLICATION SERVER's local zone, which is
    not necessarily the database session's TimeZone. Set LEGACY_TIMESTAMP_TZ (an
    IANA name such as 'America/Detroit' or 'UTC') to the zone the app servers ran
    in; without it the session TimeZone is assumed and a warning is printed.
    The value is validated against pg_timezone_names so a typo fails the migration
    instead of silently shifting every historical timestamp.
    """
    import os
    tz = (os.getenv("LEGACY_TIMESTAMP_TZ") or "").strip()
    if not tz:
        cur.execute("SELECT current_setting('TimeZone')")
        tz = cur.fetchone()[0]
        print(f"  WARNING: LEGACY_TIMESTAMP_TZ not set — interpreting legacy naive "
              f"timestamps in the DB session zone '{tz}'. If the app servers ran in "
              f"a different zone, set LEGACY_TIMESTAMP_TZ before migrating.")
        return tz
    cur.execute("SELECT 1 FROM pg_timezone_names WHERE name = %s", (tz,))
    if cur.fetchone() is None:
        raise ValueError(f"LEGACY_TIMESTAMP_TZ={tz!r} is not a valid time zone name")
    return tz


def upgrade(cur):
    # ---------------------------------------------------------------- columns --
    for table, col, typ in [
        ("steps", "security_flag", "BOOLEAN NOT NULL DEFAULT FALSE"),
        ("steps", "security_reason", "TEXT"),
        ("steps", "run_attempt", "INTEGER"),
        ("steps", "reviewer_user_id", "INTEGER"),
        ("llm_calls", "run_attempt", "INTEGER"),
        ("tool_calls", "run_attempt", "INTEGER"),
        ("runs", "attempt", "INTEGER NOT NULL DEFAULT 0"),
        ("runs", "last_attempt_ended_at", "TIMESTAMP"),
        ("job_reqs_cache", "expires_at", "TIMESTAMP"),
        ("job_searches", "location_filter_applied", "BOOLEAN NOT NULL DEFAULT TRUE"),
    ]:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}")

    # Resume-prompt-injection used to be written as a REVIEW flag even though the graph
    # never pauses after parse_resume. Move it to the security flag.
    cur.execute("""
        UPDATE steps
        SET security_flag = TRUE,
            security_reason = 'possible_prompt_injection(resume)',
            needs_human_review = FALSE,
            review_reason = NULLIF(TRIM(BOTH '; ' FROM
                REPLACE(COALESCE(review_reason, ''), 'possible_prompt_injection(resume)', '')), '')
        WHERE step_name = 'parse_resume'
          AND review_reason LIKE '%%possible_prompt_injection(resume)%%'
    """)

    # Existing trace rows predate attempts: they belong to attempt 1.
    cur.execute("UPDATE runs SET attempt = 1 WHERE attempt = 0 AND started_at IS NOT NULL")
    for t in ("steps", "llm_calls", "tool_calls"):
        cur.execute(f"UPDATE {t} SET run_attempt = 1 WHERE run_attempt IS NULL")

    # Remotive ('live') postings were inserted with last_seen_at NULL, which the
    # freshness filter treated as "fresh forever". Date them from when they were
    # stored, so old ones correctly age out; the next sighting refreshes them.
    cur.execute("""
        UPDATE job_postings
        SET last_seen_at = COALESCE(created_at, NOW()),
            fetched_at   = COALESCE(fetched_at, created_at, NOW())
        WHERE source = 'live' AND last_seen_at IS NULL
    """)
    # Remotive never geo-filters; mark its historical searches accordingly.
    cur.execute("UPDATE job_searches SET location_filter_applied = FALSE WHERE source = 'remotive'")

    # Any existing rule-based cache rows: give them a short life so the LLM gets a
    # chance to produce a better extraction.
    cur.execute("""
        UPDATE job_reqs_cache SET expires_at = NOW() + INTERVAL '1 day'
        WHERE extraction_method IS DISTINCT FROM 'llm' AND expires_at IS NULL
    """)

    # ------------------------------------------------------------ timestamptz --
    cur.execute("""
        SELECT table_name, column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND data_type = 'timestamp without time zone'
          AND table_name = ANY(%s)
    """, (list(_APP_TABLES),))
    naive_columns = cur.fetchall()
    if naive_columns:
        legacy_tz = _legacy_timezone(cur)
        for table, col in naive_columns:
            cur.execute(
                f"ALTER TABLE {table} ALTER COLUMN {col} TYPE TIMESTAMPTZ "
                f"USING {col} AT TIME ZONE %s", (legacy_tz,))

    # ------------------------------------------------------------ constraints --
    # Foreign keys the application relied on but the database didn't enforce.
    _add_constraint(cur, "runs", "runs_resume_fk",
                    "FOREIGN KEY (resume_id) REFERENCES resumes(id)")
    _add_constraint(cur, "runs", "runs_user_fk",
                    "FOREIGN KEY (user_id) REFERENCES users(id)")
    _add_constraint(cur, "resumes", "resumes_user_fk",
                    "FOREIGN KEY (user_id) REFERENCES users(id)")
    _add_constraint(cur, "job_queue", "job_queue_run_fk",
                    "FOREIGN KEY (run_id) REFERENCES runs(id)")
    _add_constraint(cur, "run_advice", "run_advice_run_fk",
                    "FOREIGN KEY (run_id) REFERENCES runs(id)")
    _add_constraint(cur, "steps", "steps_reviewer_fk",
                    "FOREIGN KEY (reviewer_user_id) REFERENCES users(id)")

    # Value checks.
    _add_constraint(cur, "runs", "runs_status_chk",
                    f"CHECK (status IN {_in(RUN_STATUSES)})")
    _add_constraint(cur, "runs", "runs_attempt_chk", "CHECK (attempt >= 0)")
    _add_constraint(cur, "job_queue", "job_queue_status_chk",
                    f"CHECK (status IN {_in(QUEUE_STATUSES)})")
    _add_constraint(cur, "job_queue", "job_queue_attempts_chk",
                    "CHECK (attempts >= 0 AND max_attempts >= 1)")
    _add_constraint(cur, "steps", "steps_final_decision_chk",
                    f"CHECK (final_decision IS NULL OR final_decision IN {_in(DECISIONS)})")
    _add_constraint(cur, "steps", "steps_score_decision_chk",
                    f"CHECK (score_decision IS NULL OR score_decision IN {_in(DECISIONS)})")
    _add_constraint(cur, "steps", "steps_review_status_chk",
                    f"CHECK (review_status IS NULL OR review_status IN {_in(DECISIONS)})")
    _add_constraint(cur, "run_rankings", "run_rankings_decision_chk",
                    f"CHECK (final_decision IS NULL OR final_decision IN {_in(DECISIONS)})")
    _add_constraint(cur, "run_rankings", "run_rankings_position_chk",
                    "CHECK (rank_position >= 1)")

    # Uniqueness. rank positions are unique per run; a job appears once per ranking
    # and once per advice set (job_id may be NULL for legacy rows — NULLs are distinct).
    _add_unique(cur, "run_rankings", "run_rankings_run_pos_uniq", "run_id, rank_position")
    _add_unique(cur, "run_rankings", "run_rankings_run_job_uniq", "run_id, job_id")
    _add_unique(cur, "run_advice", "run_advice_run_job_uniq", "run_id, job_id")

    # ---------------------------------------------------------------- indexes --
    for name, table, cols in [
        ("steps_run_order_idx", "steps", "run_id, run_attempt, step_order"),
        ("llm_calls_step_idx", "llm_calls", "step_id, id"),
        ("llm_calls_run_idx", "llm_calls", "run_id"),
        ("tool_calls_step_idx", "tool_calls", "step_id, id"),
        ("tool_calls_run_idx", "tool_calls", "run_id"),
        ("evaluations_step_idx", "evaluations", "step_id, id"),
        ("evaluations_run_idx", "evaluations", "run_id"),
        ("job_queue_claim_idx", "job_queue", "status, available_at, enqueued_at"),
        ("job_queue_heartbeat_idx", "job_queue", "status, heartbeat_at"),
        ("job_queue_run_idx", "job_queue", "run_id"),
        ("runs_user_idx", "runs", "user_id, id"),
        ("resumes_user_idx", "resumes", "user_id, id"),
    ]:
        cur.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({cols})")