"""
Data erasure and trace retention.

REDACT_TRACE_PAYLOADS (formerly REDACT_SENSITIVE) only hides fields from API RESPONSES. The database still holds
resume text, LLM prompts/responses (which embed the resume), tool I/O, retrieved
context, advice, and LangGraph checkpoints (whose state carries the full resume
text). "Not shown by the API" is not "not stored", so this module provides the
real controls:

  erase_resume(resume_id, user_id)  – user deletes a resume: its text AND every
                                      trace/derived artifact that embeds it is erased,
                                      including cached parses from ALL parser versions.
  delete_run(run_id, user_id)       – hard-delete one finished run and all its traces.
  purge_expired_traces(days)        – retention: run the idempotent scrubber on EVERY
                                      terminal run that ended more than `days` ago
                                      (prompts, responses, tool I/O, context, review
                                      comments, evaluation notes, advice, pending
                                      review, checkpoints). Metrics (tokens, cost,
                                      latency, status, scores) are kept.

Erasure is MONOTONIC (erasure_guards.py): every erased run gets an append-only
tombstone, its execution_generation is bumped (so any worker still holding the old
generation fails its next fenced write), and database triggers keep any LATE write
from a stale worker — trace rows, step context, tool payloads, suggestions, advice,
LangGraph checkpoints — from re-persisting the erased content. An erased resume
cannot be restored or rewritten, and its parse cannot be re-cached.

What this does NOT cover (deployment responsibilities, see README "Data handling"):
database-level encryption at rest, backup retention/rotation (erased data lives on
in old backups until they expire), and log retention.
"""
from database import get_connection
from logging_config import get_logger

log = get_logger(__name__)

ERASED = "[erased]"
ACTIVE_RUN_STATUSES = ("queued", "running", "retrying", "waiting_for_human")
_CHECKPOINT_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")


class ErasureConflict(Exception):
    """The data is in use by an active run and can't be erased right now."""


def _existing_checkpoint_tables(cur):
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                "AND tablename = ANY(%s)", (list(_CHECKPOINT_TABLES),))
    have = {r[0] for r in cur.fetchall()}
    return [t for t in _CHECKPOINT_TABLES if t in have]


def _tombstone_runs(cur, run_ids, reason):
    """Record the erasure FIRST (same transaction as the scrub) and supersede every
    execution generation of these runs. From the commit on, the guard triggers
    blank or drop any late write for them, and a worker still holding an old
    generation gets ExecutionLost on its next fenced write. Idempotent: an
    already-tombstoned run keeps its original tombstone (append-only)."""
    if not run_ids:
        return
    cur.execute("INSERT INTO erasure_tombstones (run_id, reason) "
                "SELECT unnest(%s::bigint[]), %s ON CONFLICT (run_id) DO NOTHING",
                (list(run_ids), reason))
    cur.execute("UPDATE runs SET execution_generation = execution_generation + 1 "
                "WHERE id = ANY(%s)", (list(run_ids),))


def is_erased(run_id):
    """True if this run's payloads were erased (tombstoned)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM erasure_tombstones WHERE run_id = %s", (run_id,))
        return cur.fetchone() is not None


def _scrub_run_payloads(cur, run_ids):
    """Null every trace field that can contain resume/job content for these runs,
    and drop their LangGraph checkpoints. Keeps the numeric/status trace intact."""
    if not run_ids:
        return
    cur.execute("UPDATE llm_calls SET prompt = NULL, response = NULL WHERE run_id = ANY(%s)",
                (run_ids,))
    cur.execute("UPDATE tool_calls SET input_json = NULL, output_json = NULL "
                "WHERE run_id = ANY(%s)", (run_ids,))
    cur.execute("UPDATE steps SET retrieved_context = NULL, review_comment = NULL "
                "WHERE run_id = ANY(%s)", (run_ids,))
    cur.execute("UPDATE evaluations SET hallucinated_claims = NULL, notes = NULL "
                "WHERE run_id = ANY(%s)", (run_ids,))
    cur.execute("UPDATE run_advice SET advice = %s WHERE run_id = ANY(%s)", (ERASED, run_ids))
    cur.execute("UPDATE runs SET pending_review = NULL WHERE id = ANY(%s)", (run_ids,))
    # R04: error text can embed resume/job content; review comments are user-authored.
    cur.execute("UPDATE llm_calls SET error_message = NULL WHERE run_id = ANY(%s)", (run_ids,))
    cur.execute("UPDATE tool_calls SET error_message = NULL WHERE run_id = ANY(%s)", (run_ids,))
    cur.execute("UPDATE steps SET error_message = NULL WHERE run_id = ANY(%s)", (run_ids,))
    cur.execute("UPDATE runs SET stop_reason = CASE WHEN stop_reason IS NULL THEN NULL "
                "ELSE '[erased]' END, goal_progress = NULL WHERE id = ANY(%s)", (run_ids,))
    cur.execute("UPDATE job_queue SET payload = payload - 'comment' - 'answer', last_error = NULL "
                "WHERE run_id = ANY(%s)", (run_ids,))
    # Agent-controller tables (migration 0010): observations can quote job titles,
    # review requests carry comments/answers, suggestions quote the resume.
    cur.execute("UPDATE agent_actions SET arguments = NULL, observation = NULL, reason = NULL, "
                "error = NULL WHERE run_id = ANY(%s)", (run_ids,))
    cur.execute("UPDATE agent_action_attempts SET observation = NULL, error = NULL "
                "WHERE run_id = ANY(%s)", (run_ids,))
    cur.execute("UPDATE review_requests SET payload = NULL, comment = NULL, answer = NULL "
                "WHERE run_id = ANY(%s)", (run_ids,))
    cur.execute("DELETE FROM resume_suggestions WHERE run_id = ANY(%s)", (run_ids,))
    threads = [str(r) for r in run_ids]
    for t in _existing_checkpoint_tables(cur):
        cur.execute(f"DELETE FROM {t} WHERE thread_id = ANY(%s)", (threads,))


def erase_resume(resume_id, user_id):
    """
    Erase a user's resume and everything derived from it. Returns False if the
    resume doesn't exist for this user. Raises ErasureConflict if a run using it is
    still active (erasing mid-run would break that run).

    The resumes row is kept (runs still reference it by FK, and the run history —
    status, cost, scores — stays auditable) but its text and name are overwritten.
    The cached parse of the text is deleted too.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT resume_text FROM resumes WHERE id = %s AND user_id = %s FOR UPDATE",
                    (resume_id, user_id))
        row = cur.fetchone()
        if not row:
            return False
        cur.execute("SELECT count(*) FROM runs WHERE resume_id = %s AND status = ANY(%s)",
                    (resume_id, list(ACTIVE_RUN_STATUSES)))
        if cur.fetchone()[0]:
            raise ErasureConflict("resume is used by a run that is still in progress")

        text = row[0] or ""
        if text and text != ERASED:
            # The cache is keyed by a STABLE content hash plus a separate version, so
            # this removes EVERY cached parse of this text — including parses made
            # by older parser/schema/model versions (derived personal data).
            from router import resume_content_hash
            h = resume_content_hash(text)
            # Hash tombstone first: a stale worker can no longer re-cache the parse.
            cur.execute("INSERT INTO erased_resume_hashes (content_hash) VALUES (%s) "
                        "ON CONFLICT (content_hash) DO NOTHING", (h,))
            cur.execute("DELETE FROM parsed_resume_cache WHERE content_hash = %s", (h,))
        cur.execute("SELECT id FROM runs WHERE resume_id = %s ORDER BY id FOR UPDATE", (resume_id,))
        run_ids = [r[0] for r in cur.fetchall()]
        _tombstone_runs(cur, run_ids, "resume_erased")
        _scrub_run_payloads(cur, run_ids)
        cur.execute("UPDATE resumes SET resume_text = %s, name = %s, is_deleted = TRUE "
                    "WHERE id = %s", (ERASED, ERASED, resume_id))
    log.info("resume erased", extra={"resume_id": resume_id})
    return True


def delete_run(run_id, user_id):
    """
    Hard-delete one of the user's runs and all of its traces. Returns False if the
    run doesn't exist for this user; raises ErasureConflict if it is still active.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status FROM runs WHERE id = %s AND user_id = %s FOR UPDATE",
                    (run_id, user_id))
        row = cur.fetchone()
        if not row:
            return False
        if row[0] in ACTIVE_RUN_STATUSES:
            raise ErasureConflict("run is still in progress; cancel it first")
        # The tombstone outlives the row: the checkpoint tables have no FK to runs,
        # so without it a stale worker could recreate this run's checkpoints.
        _tombstone_runs(cur, [run_id], "run_deleted")
        _scrub_run_payloads(cur, [run_id])       # checkpoints live outside the FK graph
        for table in ("evaluations", "llm_calls", "tool_calls", "run_rankings", "run_advice",
                      "agent_action_attempts", "agent_actions", "agent_searches", "review_requests",
                      "resume_suggestions"):
            cur.execute(f"DELETE FROM {table} WHERE run_id = %s", (run_id,))
        cur.execute("DELETE FROM steps WHERE run_id = %s", (run_id,))
        cur.execute("DELETE FROM job_searches WHERE run_id = %s", (run_id,))  # results cascade
        cur.execute("DELETE FROM job_queue WHERE run_id = %s", (run_id,))
        cur.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    log.info("run deleted", extra={"run_id": run_id})
    return True


def purge_expired_traces(retention_days, batch_size=500):
    """
    Retention: scrub sensitive payloads from EVERY terminal run that ENDED more than
    retention_days ago. Idempotent. Returns the number of runs scrubbed.
    retention_days <= 0 disables purging.

    Eligibility is (terminal status AND ended before the cutoff) — deliberately NOT
    "still has a non-NULL prompt/tool input". A run whose LLM/tool payloads happen
    to be gone can still hold review comments, evaluation notes, hallucinated
    claims, advice, a pending-review payload or LangGraph checkpoint data; the old
    EXISTS filter never selected such runs, so that data outlived the retention
    period. The scrubber is idempotent, so re-scrubbing a clean run is harmless.

    Runs already scrubbed are remembered via runs.trace_purged_at so each sweep only
    touches NEW expirations (processed in batches to keep transactions short).
    """
    if not retention_days or retention_days <= 0:
        return 0
    total = 0
    while True:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id FROM runs
                WHERE ended_at IS NOT NULL
                  AND ended_at < NOW() - make_interval(days => %s)
                  AND status <> ALL(%s)
                  AND trace_purged_at IS NULL
                ORDER BY id
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            """, (int(retention_days), list(ACTIVE_RUN_STATUSES), int(batch_size)))
            run_ids = [r[0] for r in cur.fetchall()]
            if not run_ids:
                break
            _tombstone_runs(cur, run_ids, "retention")
            _scrub_run_payloads(cur, run_ids)
            cur.execute("UPDATE runs SET trace_purged_at = NOW() WHERE id = ANY(%s)",
                        (run_ids,))
        total += len(run_ids)
        if len(run_ids) < batch_size:
            break
    if total:
        log.info("retention: scrubbed trace payloads of %d run(s)", total)
    return total