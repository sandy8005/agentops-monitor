"""
Data erasure and trace retention.

REDACT_SENSITIVE only hides fields from API RESPONSES. The database still holds
resume text, LLM prompts/responses (which embed the resume), tool I/O, retrieved
context, advice, and LangGraph checkpoints (whose state carries the full resume
text). "Not shown by the API" is not "not stored", so this module provides the
real controls:

  erase_resume(resume_id, user_id)  – user deletes a resume: its text AND every
                                      trace/derived artifact that embeds it is erased.
  delete_run(run_id, user_id)       – hard-delete one finished run and all its traces.
  purge_expired_traces(days)        – retention: strip trace payloads (prompts,
                                      responses, tool I/O, context, checkpoints) from
                                      runs that ended more than `days` ago. Metrics
                                      (tokens, cost, latency, status, scores) are kept.

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
            from router import _resume_cache_key
            cur.execute("DELETE FROM parsed_resume_cache WHERE resume_hash = %s",
                        (_resume_cache_key(text),))
        cur.execute("SELECT id FROM runs WHERE resume_id = %s", (resume_id,))
        _scrub_run_payloads(cur, [r[0] for r in cur.fetchall()])
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
        _scrub_run_payloads(cur, [run_id])       # checkpoints live outside the FK graph
        for table in ("evaluations", "llm_calls", "tool_calls", "run_rankings", "run_advice"):
            cur.execute(f"DELETE FROM {table} WHERE run_id = %s", (run_id,))
        cur.execute("DELETE FROM steps WHERE run_id = %s", (run_id,))
        cur.execute("DELETE FROM job_searches WHERE run_id = %s", (run_id,))  # results cascade
        cur.execute("DELETE FROM job_queue WHERE run_id = %s", (run_id,))
        cur.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    log.info("run deleted", extra={"run_id": run_id})
    return True


def purge_expired_traces(retention_days):
    """
    Retention: strip sensitive trace payloads from runs that ENDED more than
    retention_days ago. Idempotent. Returns the number of runs purged.
    retention_days <= 0 disables purging.
    """
    if not retention_days or retention_days <= 0:
        return 0
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id FROM runs
            WHERE ended_at IS NOT NULL
              AND ended_at < NOW() - make_interval(days => %s)
              AND status <> ALL(%s)
              AND EXISTS (SELECT 1 FROM llm_calls l WHERE l.run_id = runs.id
                          AND (l.prompt IS NOT NULL OR l.response IS NOT NULL)
                          UNION ALL
                          SELECT 1 FROM tool_calls t WHERE t.run_id = runs.id
                          AND t.input_json IS NOT NULL)
        """, (int(retention_days), list(ACTIVE_RUN_STATUSES)))
        run_ids = [r[0] for r in cur.fetchall()]
        _scrub_run_payloads(cur, run_ids)
    if run_ids:
        log.info("retention: purged trace payloads of %d run(s)", len(run_ids))
    return len(run_ids)