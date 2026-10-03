"""
Database-enforced, MONOTONIC data erasure.

Erasure used to be a one-shot scrub: UPDATE ... SET prompt = NULL etc. Anything that
wrote AFTER the scrub — a superseded worker generation still finishing an LLM call,
a LangGraph checkpoint flush, a late advice/suggestion insert — re-persisted the
data the user had just erased. Application-level fencing (execution_generation)
covers the writes that go through agent_store, but not llm_calls/tool_calls trace
rows, router step context, or the checkpointer, which writes through its own
connection.

The rule is therefore enforced where every writer has to pass: the database.

  erasure_tombstones(run_id)        one row per erased run. Append-only (UPDATE and
                                    DELETE raise), no FK, so it outlives a hard-
                                    deleted run. Reasons: resume_erased /
                                    run_deleted / retention.
  erased_resume_hashes(hash)        content hashes of erased resume texts: a stale
                                    worker cannot re-populate parsed_resume_cache.
                                    Cleared only by a NEW upload of the same text
                                    (fresh, explicit consent to store it).

BEFORE INSERT OR UPDATE triggers on every payload-bearing table consult the
tombstones and, for a tombstoned run:
  * null the sensitive columns while keeping metrics (tokens, cost, status) so the
    cost ledger stays complete           (llm_calls, tool_calls, steps, ...)
  * or drop the row entirely              (resume_suggestions, checkpoints)
Resumes cannot be un-deleted and an erased resume's text cannot be rewritten.

install_app_guards() runs in migration 0015; install_checkpoint_guards() runs in
the same migration when LangGraph's tables already exist and after every
checkpointing.setup_schema(), so the checkpoint tables are covered on fresh
databases too. Both are idempotent.
"""

TOMBSTONE_REASONS = ("resume_erased", "run_deleted", "retention")

# (table, mode, columns). mode:
#   null  -> set the listed columns to NULL
#   mark  -> set the listed (text) columns to '[erased]'
#   skip  -> do not write the row at all
APP_GUARDS = (
    ("llm_calls", "null", ("prompt", "response", "error_message")),
    ("tool_calls", "null", ("input_json", "output_json", "error_message")),
    ("steps", "null", ("retrieved_context", "review_comment", "error_message")),
    ("evaluations", "null", ("hallucinated_claims", "notes")),
    ("run_advice", "mark", ("advice",)),
    ("agent_actions", "null", ("arguments", "observation", "reason", "error")),
    ("agent_action_attempts", "null", ("observation", "error")),
    ("review_requests", "null", ("payload", "comment", "answer")),
    ("resume_suggestions", "skip", ()),
)
CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")

_FUNCTIONS_SQL = r"""
CREATE OR REPLACE FUNCTION erasure_is_tombstoned(p_run_id BIGINT) RETURNS BOOLEAN
LANGUAGE sql STABLE AS $$
    SELECT p_run_id IS NOT NULL
       AND EXISTS (SELECT 1 FROM erasure_tombstones WHERE run_id = p_run_id)
$$;

CREATE OR REPLACE FUNCTION erasure_tombstones_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'erasure_tombstones is append-only (erasure is monotonic)'
        USING ERRCODE = 'restrict_violation';
END $$;

-- Generic payload guard. TG_ARGV[0] = mode (null | mark | skip), TG_ARGV[1..] = columns.
CREATE OR REPLACE FUNCTION erasure_guard_payload() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    patch JSONB := '{}'::jsonb;
    i INT;
BEGIN
    IF NOT erasure_is_tombstoned(NEW.run_id::bigint) THEN
        RETURN NEW;
    END IF;
    IF TG_ARGV[0] = 'skip' THEN
        RETURN NULL;
    END IF;
    FOR i IN 1 .. TG_NARGS - 1 LOOP
        IF TG_ARGV[0] = 'mark' THEN
            patch := patch || jsonb_build_object(TG_ARGV[i], '[erased]');
        ELSE
            patch := patch || jsonb_build_object(TG_ARGV[i], NULL);
        END IF;
    END LOOP;
    NEW := jsonb_populate_record(NEW, patch);
    RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION erasure_guard_runs() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF erasure_is_tombstoned(NEW.id::bigint) THEN
        NEW.pending_review := NULL;
        NEW.goal_progress := NULL;
        IF NEW.stop_reason IS NOT NULL THEN
            NEW.stop_reason := '[erased]';
        END IF;
    END IF;
    RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION erasure_guard_job_queue() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF erasure_is_tombstoned(NEW.run_id::bigint) THEN
        NEW.payload := COALESCE(NEW.payload, '{}'::jsonb) - 'comment' - 'answer';
        NEW.last_error := NULL;
    END IF;
    RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION erasure_guard_parse_cache() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM erased_resume_hashes WHERE content_hash = NEW.content_hash) THEN
        RETURN NULL;
    END IF;
    RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION erasure_guard_resumes() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.is_deleted AND NOT NEW.is_deleted THEN
        RAISE EXCEPTION 'resume % was erased and cannot be restored', OLD.id
            USING ERRCODE = 'restrict_violation';
    END IF;
    IF OLD.is_deleted AND (NEW.resume_text IS DISTINCT FROM OLD.resume_text
                           OR NEW.name IS DISTINCT FROM OLD.name) THEN
        RAISE EXCEPTION 'resume % was erased; its content cannot be rewritten', OLD.id
            USING ERRCODE = 'restrict_violation';
    END IF;
    RETURN NEW;
END $$;
"""

_CHECKPOINT_FUNCTION_SQL = r"""
CREATE OR REPLACE FUNCTION erasure_guard_checkpoint() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    -- Agent runs use thread_id = str(run_id).
    IF NEW.thread_id ~ '^[0-9]{1,18}$'
       AND erasure_is_tombstoned(NEW.thread_id::bigint) THEN
        RETURN NULL;
    END IF;
    RETURN NEW;
END $$;
"""


def _trigger(cur, table, name, function, args=()):
    cur.execute(f"DROP TRIGGER IF EXISTS {name} ON {table}")
    arglist = ", ".join("'" + a.replace("'", "''") + "'" for a in args)
    cur.execute(f"CREATE TRIGGER {name} BEFORE INSERT OR UPDATE ON {table} "
                f"FOR EACH ROW EXECUTE FUNCTION {function}({arglist})")


def install_app_guards(cur):
    """Tombstone tables, functions and triggers on the application tables."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS erasure_tombstones (
            run_id BIGINT PRIMARY KEY,
            reason TEXT NOT NULL
                CHECK (reason IN ('resume_erased', 'run_deleted', 'retention')),
            erased_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS erased_resume_hashes (
            content_hash TEXT PRIMARY KEY,
            erased_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    cur.execute(_FUNCTIONS_SQL)
    cur.execute("DROP TRIGGER IF EXISTS erasure_tombstones_append_only ON erasure_tombstones")
    cur.execute("CREATE TRIGGER erasure_tombstones_append_only BEFORE UPDATE OR DELETE "
                "ON erasure_tombstones FOR EACH ROW "
                "EXECUTE FUNCTION erasure_tombstones_append_only()")
    for table, mode, cols in APP_GUARDS:
        _trigger(cur, table, f"{table}_erasure_guard", "erasure_guard_payload", (mode,) + cols)
    _trigger(cur, "runs", "runs_erasure_guard", "erasure_guard_runs")
    _trigger(cur, "job_queue", "job_queue_erasure_guard", "erasure_guard_job_queue")
    _trigger(cur, "parsed_resume_cache", "parsed_resume_cache_erasure_guard",
             "erasure_guard_parse_cache")
    cur.execute("DROP TRIGGER IF EXISTS resumes_erasure_guard ON resumes")
    cur.execute("CREATE TRIGGER resumes_erasure_guard BEFORE UPDATE ON resumes "
                "FOR EACH ROW EXECUTE FUNCTION erasure_guard_resumes()")


def install_checkpoint_guards(cur):
    """Triggers on LangGraph's checkpoint tables (only those that exist). Requires
    install_app_guards() to have run (it creates erasure_is_tombstoned)."""
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = current_schema() "
                "AND tablename = ANY(%s)", (list(CHECKPOINT_TABLES),))
    rows = cur.fetchall()
    have = {(r["tablename"] if isinstance(r, dict) else r[0]) for r in rows}
    if not have:
        return []
    cur.execute(_CHECKPOINT_FUNCTION_SQL)
    installed = []
    for t in CHECKPOINT_TABLES:
        if t in have:
            _trigger(cur, t, f"{t}_erasure_guard", "erasure_guard_checkpoint")
            installed.append(t)
    return installed


def missing_guards(cur):
    """Names of expected guard triggers that are NOT installed (empty = all good).
    Checkpoint tables are only expected once they exist."""
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = current_schema()")
    tables = {(r["tablename"] if isinstance(r, dict) else r[0]) for r in cur.fetchall()}
    expected = [f"{t}_erasure_guard" for t, _m, _c in APP_GUARDS]
    expected += ["runs_erasure_guard", "job_queue_erasure_guard",
                 "parsed_resume_cache_erasure_guard", "resumes_erasure_guard",
                 "erasure_tombstones_append_only"]
    expected += [f"{t}_erasure_guard" for t in CHECKPOINT_TABLES if t in tables]
    cur.execute("SELECT tgname FROM pg_trigger WHERE NOT tgisinternal")
    have = {(r["tgname"] if isinstance(r, dict) else r[0]) for r in cur.fetchall()}
    return [n for n in expected if n not in have]