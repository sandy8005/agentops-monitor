"""
Persistence for the controller loop. Output writes are FENCED by
runs.execution_generation (R28): a worker that lost the run raises ExecutionLost
instead of overwriting the newer execution's results.
"""
import json

from database import get_connection
from timeutil import utcnow
from sanitize import redact_secrets


class ExecutionLost(RuntimeError):
    pass


def _check_generation(cur, run_id, generation, lock="FOR SHARE"):
    cur.execute(f"SELECT execution_generation FROM runs WHERE id = %s {lock}", (run_id,))
    row = cur.fetchone()
    if not row or int(row[0]) != int(generation):
        raise ExecutionLost(f"run {run_id}: generation {generation} superseded "
                            f"(current {row[0] if row else 'missing'})")


def begin_execution(run_id, new_attempt):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE runs SET status = 'running',
                started_at = COALESCE(started_at, NOW()),
                last_attempt_ended_at = COALESCE(ended_at, last_attempt_ended_at),
                ended_at = NULL,
                attempt = CASE WHEN %s OR attempt = 0 THEN attempt + 1 ELSE attempt END,
                execution_generation = execution_generation + 1
            WHERE id = %s
            RETURNING execution_generation
        """, (bool(new_attempt), run_id))
        row = cur.fetchone()
        if not row:
            raise ExecutionLost(f"run {run_id} does not exist")
        return int(row[0])


def load_run_config(run_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT goal_json, resume_id, user_id, cancel_requested, status "
                    "FROM runs WHERE id = %s", (run_id,))
        row = cur.fetchone()
    if not row:
        return None
    goal = row[0] if isinstance(row[0], dict) else (json.loads(row[0]) if row[0] else None)
    return {"goal": goal, "resume_id": row[1], "user_id": row[2],
            "cancel_requested": bool(row[3]), "status": row[4]}


def is_cancel_requested(run_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT cancel_requested FROM runs WHERE id = %s", (run_id,))
        row = cur.fetchone()
        return bool(row and row[0])


def run_usage(run_id):
    """Elapsed time, reserved LLM calls, KNOWN estimated cost, and the count of
    calls with unknown cost (R15 — unknown is never summed as zero)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT EXTRACT(EPOCH FROM (NOW() - COALESCE(started_at, NOW()))),
                   llm_calls_reserved, llm_call_budget
            FROM runs WHERE id = %s
        """, (run_id,))
        row = cur.fetchone() or (0, 0, None)
        cur.execute("""
            SELECT COALESCE(SUM(cost_usd) FILTER (WHERE cost_usd IS NOT NULL), 0),
                   COUNT(*) FILTER (WHERE cost_usd IS NULL AND status = 'success')
            FROM llm_calls WHERE run_id = %s
        """, (run_id,))
        cost = cur.fetchone() or (0, 0)
    return {"elapsed_seconds": float(row[0] or 0), "llm_calls_reserved": int(row[1] or 0),
            "llm_call_budget": row[2], "known_cost_usd": float(cost[0] or 0),
            "unknown_cost_calls": int(cost[1] or 0)}


def reserve_llm_call(run_id, default_budget):
    """Atomically reserve ONE request against the run-wide budget (R16)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE runs SET llm_calls_reserved = llm_calls_reserved + 1
            WHERE id = %s AND llm_calls_reserved < COALESCE(llm_call_budget, %s)
            RETURNING llm_calls_reserved
        """, (run_id, int(default_budget)))
        return cur.fetchone() is not None


def set_controller_mode(run_id, generation, mode, progress):
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation)
        cur.execute("UPDATE runs SET controller_mode = %s, goal_progress = %s WHERE id = %s",
                    (mode, json.dumps(progress, default=str), run_id))


def get_action(run_id, iteration):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT action, arguments, reason, decided_by, status, observation, error
                       FROM agent_actions WHERE run_id = %s AND iteration = %s""",
                    (run_id, iteration))
        row = cur.fetchone()
    if not row:
        return None

    def _j(v):
        return v if (v is None or isinstance(v, (dict, list))) else json.loads(v)
    return {"action": row[0], "arguments": _j(row[1]) or {}, "reason": row[2],
            "decided_by": row[3], "status": row[4], "observation": _j(row[5]),
            "error": row[6]}


def record_proposed_action(run_id, generation, iteration, action, arguments, reason, decided_by):
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation)
        cur.execute("""
            INSERT INTO agent_actions (run_id, iteration, execution_generation, action,
                                       arguments, reason, decided_by, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'proposed')
            ON CONFLICT (run_id, iteration) DO NOTHING
        """, (run_id, iteration, generation, action, json.dumps(arguments, default=str),
              (reason or "")[:500], decided_by))


def record_action_outcome(run_id, generation, iteration, status, observation=None,
                          error=None, step_id=None):
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation)
        cur.execute("""
            UPDATE agent_actions SET status = %s, observation = %s, error = %s,
                   step_id = COALESCE(%s, step_id), finished_at = %s
            WHERE run_id = %s AND iteration = %s AND status = 'proposed'
        """, (status, json.dumps(observation, default=str) if observation is not None else None,
              redact_secrets(error, 1000) if error else None, step_id, utcnow(),
              run_id, iteration))


def list_actions(run_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT iteration, action, arguments, reason, decided_by, status,
                              observation, error, created_at, finished_at
                       FROM agent_actions WHERE run_id = %s ORDER BY iteration""", (run_id,))
        rows = cur.fetchall()
    return [{"iteration": r[0], "action": r[1], "arguments": r[2], "reason": r[3],
             "decided_by": r[4], "status": r[5], "observation": r[6], "error": r[7],
             "created_at": r[8].isoformat() if r[8] else None,
             "finished_at": r[9].isoformat() if r[9] else None} for r in rows]


def get_search(run_id, provider, query_norm):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT provider_status, new_jobs, duplicates, eligible_jobs,
                              rejection_summary, iteration
                       FROM agent_searches WHERE run_id = %s AND provider = %s AND query_norm = %s""",
                    (run_id, provider, query_norm))
        row = cur.fetchone()
    if not row:
        return None
    return {"provider_status": row[0], "new_jobs": row[1], "duplicates": row[2],
            "eligible_jobs": row[3], "rejection_summary": row[4], "iteration": row[5]}


def record_search(run_id, generation, iteration, provider, query_norm, location, obs):
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation)
        cur.execute("""
            INSERT INTO agent_searches (run_id, provider, query_norm, location, iteration,
                provider_status, new_jobs, duplicates, eligible_jobs, rejection_summary)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, provider, query_norm) DO NOTHING
        """, (run_id, provider, query_norm, location or None, iteration,
              obs.get("provider_status"), obs.get("new_jobs", 0), obs.get("duplicates", 0),
              obs.get("eligible_jobs", 0), json.dumps(obs.get("rejection_summary") or {})))


def load_postings(job_ids):
    """Descriptions are loaded on demand, never kept in the checkpoint (R14)."""
    ids = [int(i) for i in job_ids]
    if not ids:
        return {}
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT id, title, company, description, location, work_mode,
                              employment_type, source, external_id, apply_url
                       FROM job_postings WHERE id = ANY(%s)""", (ids,))
        rows = cur.fetchall()
    return {r[0]: {"id": r[0], "title": r[1], "company": r[2], "description": r[3] or "",
                   "location": r[4], "work_mode": r[5], "employment_type": r[6],
                   "source": r[7], "external_id": r[8], "apply_url": r[9]} for r in rows}


def qualified_job_ids(run_id, qualifying_decisions):
    """BACKEND-VERIFIED progress from persisted steps; pending reviews don't count."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT (retrieved_context->>'job_id')::int
            FROM steps
            WHERE run_id = %s
              AND retrieved_context ? 'job_id'
              AND retrieved_context->>'job_id' ~ '^[0-9]+$'
              AND final_decision = ANY(%s)
              AND status = 'success'
              AND NOT (COALESCE(needs_human_review, FALSE) AND review_status IS NULL)
        """, (run_id, list(qualifying_decisions)))
        return sorted(r[0] for r in cur.fetchall())


def create_review_request(run_id, generation, review_id, kind, payload, step_id=None, job_id=None):
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation, lock="FOR UPDATE")
        cur.execute("""UPDATE review_requests SET status = 'superseded'
                       WHERE run_id = %s AND status IN ('pending', 'submitted')
                         AND review_id <> %s""", (run_id, review_id))
        cur.execute("""
            INSERT INTO review_requests (review_id, run_id, kind, step_id, job_id, payload)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (review_id) DO NOTHING
        """, (review_id, run_id, kind, step_id, job_id, json.dumps(payload, default=str)))


def review_status(review_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT status, decision, answer, comment, reviewer_user_id, reviewer, run_id
                       FROM review_requests WHERE review_id = %s""", (review_id,))
        row = cur.fetchone()
    if not row:
        return None
    return {"status": row[0], "decision": row[1], "answer": row[2], "comment": row[3],
            "reviewer_user_id": row[4], "reviewer": row[5], "run_id": row[6]}


def mark_review_consumed(review_id, generation, run_id):
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation)
        cur.execute("""UPDATE review_requests SET status = 'consumed', consumed_at = NOW()
                       WHERE review_id = %s AND status = 'submitted' RETURNING review_id""",
                    (review_id,))
        return cur.fetchone() is not None


def persist_rankings(run_id, generation, ranked):
    """ONE fenced transaction; raises on failure (R03)."""
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation, lock="FOR UPDATE")
        cur.execute("DELETE FROM run_rankings WHERE run_id = %s", (run_id,))
        for pos, r in enumerate(ranked or [], start=1):
            cur.execute("""
                INSERT INTO run_rankings (run_id, job_id, rank_position, title, company,
                                          score, final_decision, apply_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, r.get("job_id"), pos, r.get("title"), r.get("company"),
                  r.get("score"), r.get("final_decision") or r.get("decision"),
                  r.get("apply_url")))


def persist_advice(run_id, generation, job_id, title, advice_text, suggestions,
                   resume_id, resume_hash):
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation, lock="FOR UPDATE")
        cur.execute("DELETE FROM run_advice WHERE run_id = %s AND job_id = %s", (run_id, job_id))
        cur.execute("INSERT INTO run_advice (run_id, job_id, title, advice) VALUES (%s, %s, %s, %s)",
                    (run_id, job_id, title, advice_text))
        cur.execute("DELETE FROM resume_suggestions WHERE run_id = %s AND job_id = %s",
                    (run_id, job_id))
        for pos, sg in enumerate(suggestions, start=1):
            cur.execute("""
                INSERT INTO resume_suggestions (run_id, job_id, resume_id, resume_hash, position,
                    kind, original_text, suggested_text, reason, evidence, method, status,
                    validation_notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, job_id, resume_id, resume_hash, pos, sg["kind"], sg.get("original_text"),
                  sg["suggested_text"], sg["reason"], json.dumps(sg.get("evidence") or []),
                  sg["method"], sg["status"], sg.get("validation_notes")))


def finalize(run_id, generation, status, stop_reason, error_code, progress):
    code = error_code.value if hasattr(error_code, "value") else error_code
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation, lock="FOR UPDATE")
        cur.execute("""
            UPDATE runs SET ended_at = NOW(), status = %s, stop_reason = %s, error_code = %s,
                pending_review = NULL, goal_progress = %s,
                total_tokens = (SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0)
                                FROM llm_calls WHERE run_id = %s),
                total_cost = (SELECT SUM(cost_usd) FROM llm_calls WHERE run_id = %s)
            WHERE id = %s
        """, (status, redact_secrets(stop_reason, 500), code,
              json.dumps(progress, default=str), run_id, run_id, run_id))
        cur.execute("""UPDATE review_requests SET status = 'superseded'
                       WHERE run_id = %s AND status IN ('pending', 'submitted')""", (run_id,))


def mark_waiting(run_id, generation, payload):
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation, lock="FOR UPDATE")
        cur.execute("UPDATE runs SET status = 'waiting_for_human', pending_review = %s WHERE id = %s",
                    (json.dumps(payload, default=str), run_id))