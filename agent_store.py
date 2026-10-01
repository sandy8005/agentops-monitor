"""
Persistence for the autonomous controller loop.

Every function here is a small, explicit transaction. Writes that record agent
OUTPUT (actions, searches, rankings, suggestions, final status) are FENCED by the
run's execution_generation: they run inside a transaction that first locks the run
row (FOR SHARE / FOR UPDATE) and checks the generation. A worker that lost the run
— lease reclaimed, lock connection dropped — raises ExecutionLost instead of
overwriting the newer execution's results (R28).

Kept separate from the loop logic so the loop is unit-testable with an in-memory
fake (tests/test_agent_review_fixes.py).
"""
import json

from database import get_connection
from timeutil import utcnow
from sanitize import redact_secrets


class ExecutionLost(RuntimeError):
    """This worker's execution generation is no longer the run's current one."""


def _check_generation(cur, run_id, generation, lock="FOR SHARE"):
    cur.execute(f"SELECT execution_generation FROM runs WHERE id = %s {lock}", (run_id,))
    row = cur.fetchone()
    if not row or int(row[0]) != int(generation):
        raise ExecutionLost(f"run {run_id}: generation {generation} superseded "
                            f"(current {row[0] if row else 'missing'})")


# ------------------------------------------------------------------ lifecycle --

# Close the run's open execution interval: add (NOW() - execution_started_at) to
# active_runtime_seconds and clear the marker. Used whenever execution stops being
# ACTIVE — paused for a human, finalized, or handed back after a stale resume — so
# time spent waiting is never charged to the runtime limit. Idempotent (a closed
# interval adds 0).
CLOSE_EXECUTION_INTERVAL_SQL = """
    active_runtime_seconds = active_runtime_seconds + COALESCE(
        GREATEST(0, EXTRACT(EPOCH FROM (NOW() - execution_started_at)))::double precision, 0),
    execution_started_at = NULL"""


# Charge for a DEAD worker's still-open interval (worker crashed, lock lost): up to
# its last heartbeat plus a grace period, capped at NOW(). We cannot know exactly
# when it stopped; the last heartbeat is the latest moment it was provably alive,
# and the grace (EXECUTION_HEARTBEAT_GRACE_SECONDS, default 45s, > the 30s
# heartbeat interval) covers the time between heartbeats. Without any heartbeat
# the interval is charged to NOW() (fail closed).
DEAD_INTERVAL_SQL = """
    COALESCE(GREATEST(0, EXTRACT(EPOCH FROM (
        LEAST(NOW(), COALESCE(execution_heartbeat_at + make_interval(secs => %(grace)s),
                              NOW()))
        - execution_started_at)))::double precision, 0)"""


def begin_execution(run_id, new_attempt):
    """Mark the run running, open a new ACTIVE execution interval and take a NEW
    execution generation, in one statement. Returns the generation this worker must
    present on every guarded write.

    If the previous interval was never closed (the worker died mid-execution), it is
    charged up to that worker's last heartbeat + grace (DEAD_INTERVAL_SQL). Open
    cost reservations of earlier generations become 'abandoned': their requests
    may have been billed, so they keep counting against the USD cap (and are
    reported as unknown cost) but can no longer be settled as in-flight."""
    from settings import settings
    grace = float(settings.execution_heartbeat_grace_seconds)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            UPDATE runs SET status = 'running',
                started_at = COALESCE(started_at, NOW()),
                last_attempt_ended_at = COALESCE(ended_at, last_attempt_ended_at),
                ended_at = NULL,
                attempt = CASE WHEN %(new)s OR attempt = 0 THEN attempt + 1 ELSE attempt END,
                execution_generation = execution_generation + 1,
                active_runtime_seconds = active_runtime_seconds + {DEAD_INTERVAL_SQL},
                execution_started_at = NOW(),
                execution_heartbeat_at = NOW()
            WHERE id = %(run)s
            RETURNING execution_generation
        """, {"new": bool(new_attempt), "run": run_id, "grace": grace})
        row = cur.fetchone()
        if not row:
            raise ExecutionLost(f"run {run_id} does not exist")
        gen = int(row[0])
        _abandon_open_reservations(cur, run_id, before_generation=gen)
        return gen


def _abandon_open_reservations(cur, run_id, before_generation=None):
    """open -> abandoned (still counted against the cap); keep runs.cost_reserved_usd
    equal to the sum of OPEN reservations."""
    cur.execute("""
        UPDATE llm_cost_reservations SET status = 'abandoned', settled_at = NOW()
        WHERE run_id = %s AND status = 'open'
          AND (%s::int IS NULL OR execution_generation < %s::int)
    """, (run_id, before_generation, before_generation))
    cur.execute("""UPDATE runs SET cost_reserved_usd = COALESCE((
                       SELECT SUM(amount_usd) FROM llm_cost_reservations
                       WHERE run_id = %s AND status = 'open'), 0)
                   WHERE id = %s""", (run_id, run_id))


def load_run_config(run_id):
    """(goal_json, resume_id, user_id, cancel_requested, status)."""
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


# ------------------------------------------------------------------- cost ----

def cost_summary(cur, run_id):
    """THE definition of a run's spend, shared by the guard, the reservation check,
    finalization and the API.

      known_cost_usd          priced calls (provider usage known)
      unknown_cost_calls      calls whose cost is unknown (usage missing, failed
                              after dispatch) + abandoned reservations
      unknown_cost_bound_usd  conservative upper bound of those unknown calls
      unbounded_unknown       unknown calls with NO bound (legacy rows) — a cap
                              cannot be proven while any exist (fail closed)
      reserved_open_usd       requests in flight right now
      committed_usd           known + unknown bound + open reservations: what a
                              hard cap is enforced against
    """
    cur.execute("""
        SELECT COALESCE(SUM(cost_usd) FILTER (WHERE cost_status = 'priced'), 0),
               COUNT(*) FILTER (WHERE cost_status = 'unknown'),
               COALESCE(SUM(cost_upper_bound_usd) FILTER (WHERE cost_status = 'unknown'), 0),
               COUNT(*) FILTER (WHERE cost_status = 'unknown' AND cost_upper_bound_usd IS NULL)
        FROM llm_calls WHERE run_id = %s
    """, (run_id,))
    known, unknown_calls, unknown_bound, unbounded = cur.fetchone()
    cur.execute("""
        SELECT COALESCE(SUM(amount_usd) FILTER (WHERE status = 'open'), 0),
               COUNT(*) FILTER (WHERE status = 'abandoned'),
               COALESCE(SUM(amount_usd) FILTER (WHERE status = 'abandoned'), 0),
               COUNT(*) FILTER (WHERE status IN ('open', 'abandoned') AND amount_usd IS NULL)
        FROM llm_cost_reservations WHERE run_id = %s
    """, (run_id,))
    open_amt, abandoned_n, abandoned_amt, unbounded_res = cur.fetchone()
    known = float(known or 0)
    bound = float(unknown_bound or 0) + float(abandoned_amt or 0)
    open_amt = float(open_amt or 0)
    return {"known_cost_usd": known,
            "unknown_cost_calls": int(unknown_calls or 0) + int(abandoned_n or 0),
            "unknown_cost_bound_usd": bound,
            "unbounded_unknown": int(unbounded or 0) + int(unbounded_res or 0),
            "reserved_open_usd": open_amt,
            "committed_usd": known + bound + open_amt}


def run_usage(run_id):
    """Durable usage snapshot used by the guard.

      active_runtime_seconds  time the agent actually EXECUTED: closed intervals +
                              the currently open one. Human-review waits and queue
                              backoff are not included. This is what the runtime
                              limit is enforced against.
      elapsed_seconds         wall-clock since the run first started (display only).
      + every field of cost_summary().
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT EXTRACT(EPOCH FROM (NOW() - COALESCE(started_at, NOW()))),
                   active_runtime_seconds + COALESCE(
                       GREATEST(0, EXTRACT(EPOCH FROM (NOW() - execution_started_at)))::double precision, 0),
                   llm_calls_reserved, llm_call_budget
            FROM runs WHERE id = %s
        """, (run_id,))
        row = cur.fetchone() or (0, 0, 0, None)
        cost = cost_summary(cur, run_id)
    return {"elapsed_seconds": float(row[0] or 0), "active_runtime_seconds": float(row[1] or 0),
            "llm_calls_reserved": int(row[2] or 0), "llm_call_budget": row[3], **cost}


def reserve_llm_call(run_id, generation, default_budget, projected_usd=None,
                     max_cost_usd=0.0, operation=None):
    """Atomically reserve ONE provider request — its request count AND its maximum
    possible dollar cost — before it is dispatched.

    In ONE transaction, under the run row lock:
      * the worker must still own the run (execution_generation) — a superseded
        worker gets ExecutionLost and makes no request;
      * the request-count budget must have room (else returns None);
      * with a USD cap: no unbounded unknown spend may exist (CostUnknown), the
        request must have a known maximum cost (CostUnknown), and
            known + unknown upper bounds + open reservations + projected <= cap
        (else CostLimitReached — the request is refused BEFORE it is made).
    Returns {"reservation_id", "amount_usd"}. The reservation is settled in the
    same transaction that records the call (llm._log_llm_attempt); if the worker
    dies first it stays counted (open, then abandoned) — fail closed."""
    from llm import CostUnknown, CostLimitReached
    cap = float(max_cost_usd or 0)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT execution_generation, llm_calls_reserved,
                              COALESCE(llm_call_budget, %s)
                       FROM runs WHERE id = %s FOR UPDATE""", (int(default_budget), run_id))
        row = cur.fetchone()
        if not row or int(row[0]) != int(generation):
            raise ExecutionLost(f"run {run_id}: generation {generation} superseded "
                                f"(current {row[0] if row else 'missing'}) — no model call")
        if int(row[1]) >= int(row[2]):
            return None
        if cap > 0:
            if projected_usd is None:
                raise CostUnknown("cost_unknown: the request's maximum cost cannot be "
                                  f"computed; the ${cap} cost limit cannot be enforced")
            spent = cost_summary(cur, run_id)
            if spent["unbounded_unknown"]:
                raise CostUnknown(f"cost_unknown: {spent['unbounded_unknown']} earlier call(s) "
                                  f"have unknown, unbounded cost; the ${cap} cost limit "
                                  f"cannot be enforced")
            if spent["committed_usd"] + float(projected_usd) > cap + 1e-12:
                raise CostLimitReached(
                    f"cost limit: ${spent['committed_usd']:.6f} committed + "
                    f"${float(projected_usd):.6f} for this request would exceed ${cap}")
        cur.execute("UPDATE runs SET llm_calls_reserved = llm_calls_reserved + 1, "
                    "cost_reserved_usd = cost_reserved_usd + %s WHERE id = %s",
                    (float(projected_usd or 0), run_id))
        cur.execute("""INSERT INTO llm_cost_reservations (run_id, execution_generation,
                           operation, amount_usd)
                       VALUES (%s, %s, %s, %s) RETURNING id""",
                    (run_id, int(generation), (operation or "")[:80], projected_usd))
        return {"reservation_id": int(cur.fetchone()[0]),
                "amount_usd": None if projected_usd is None else float(projected_usd)}


def set_controller_mode(run_id, generation, mode, progress):
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation)
        cur.execute("UPDATE runs SET controller_mode = %s, goal_progress = %s WHERE id = %s",
                    (mode, json.dumps(progress, default=str), run_id))


# -------------------------------------------------------------------- actions --

def _j(v):
    return v if (v is None or isinstance(v, (dict, list))) else json.loads(v)


def get_action(run_id, iteration):
    """The recorded controller DECISION for (run, iteration), or None. A decision is
    reused on replay (the model is never re-asked); each execution of it is a
    separate row in agent_action_attempts."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT action, arguments, reason, decided_by, status, observation, error,
                              id, execution_generation, attempt_count
                       FROM agent_actions WHERE run_id = %s AND iteration = %s""",
                    (run_id, iteration))
        row = cur.fetchone()
    if not row:
        return None
    return {"action": row[0], "arguments": _j(row[1]) or {}, "reason": row[2],
            "decided_by": row[3], "status": row[4], "observation": _j(row[5]),
            "error": row[6], "action_id": row[7], "decision_generation": row[8],
            "attempt_count": int(row[9] or 0)}


def record_proposed_action(run_id, generation, iteration, action, arguments, reason, decided_by):
    """Insert the DECISION. ON CONFLICT DO NOTHING: a replayed iteration keeps the
    ORIGINAL decision; executions are recorded as attempts, never as new decisions."""
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


def _ensure_attempt(cur, run_id, generation, iteration, replayed):
    """Return (action_id, attempt_number) of THIS generation's attempt at the
    decision for (run, iteration), creating it if needed. One attempt per
    (decision, execution generation): a new worker generation executing the same
    decision is a new, numbered attempt."""
    cur.execute("SELECT id, attempt_count FROM agent_actions WHERE run_id = %s AND iteration = %s "
                "FOR UPDATE", (run_id, iteration))
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"run {run_id}: no recorded decision for iteration {iteration}")
    action_id, count = row[0], int(row[1] or 0)
    cur.execute("SELECT attempt_number FROM agent_action_attempts "
                "WHERE action_id = %s AND execution_generation = %s", (action_id, generation))
    existing = cur.fetchone()
    if existing:
        return action_id, int(existing[0])
    n = count + 1
    cur.execute("""
        INSERT INTO agent_action_attempts (action_id, run_id, iteration, execution_generation,
            run_attempt, attempt_number, decision_replayed, status)
        VALUES (%s, %s, %s, %s, (SELECT attempt FROM runs WHERE id = %s), %s, %s, 'running')
    """, (action_id, run_id, iteration, generation, run_id, n, bool(replayed)))
    cur.execute("UPDATE agent_actions SET attempt_count = %s, last_execution_generation = %s "
                "WHERE id = %s", (n, generation, action_id))
    return action_id, n


def begin_action_attempt(run_id, generation, iteration, replayed=False):
    """Durably mark that THIS generation is about to execute the decision. If the
    worker dies mid-tool, the attempt stays 'running' in the trace instead of
    vanishing. Returns the attempt number."""
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation)
        return _ensure_attempt(cur, run_id, generation, iteration, replayed)[1]


def record_action_outcome(run_id, generation, iteration, status, observation=None,
                          error=None, step_id=None, replayed=False):
    """Record the outcome of THIS generation's attempt, and mirror it onto the
    decision row as its LATEST outcome. Earlier attempts are never overwritten."""
    obs_json = json.dumps(observation, default=str) if observation is not None else None
    err = redact_secrets(error, 1000) if error else None
    now = utcnow()
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation)
        action_id, _n = _ensure_attempt(cur, run_id, generation, iteration, replayed)
        cur.execute("""
            UPDATE agent_action_attempts SET status = %s, observation = %s, error = %s,
                   step_id = COALESCE(%s, step_id), finished_at = %s
            WHERE action_id = %s AND execution_generation = %s
        """, (status, obs_json, err, step_id, now, action_id, generation))
        cur.execute("""
            UPDATE agent_actions SET status = %s, observation = %s, error = %s,
                   step_id = COALESCE(%s, step_id), finished_at = %s
            WHERE id = %s AND last_execution_generation = %s
        """, (status, obs_json, err, step_id, now, action_id, generation))


def list_actions(run_id):
    """Every decision with its execution attempts nested, oldest first."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT id, iteration, action, arguments, reason, decided_by, status,
                              observation, error, created_at, finished_at,
                              execution_generation, attempt_count
                       FROM agent_actions WHERE run_id = %s ORDER BY iteration""", (run_id,))
        rows = cur.fetchall()
        cur.execute("""SELECT action_id, attempt_number, execution_generation, run_attempt,
                              decision_replayed, status, observation, error, step_id,
                              started_at, finished_at
                       FROM agent_action_attempts WHERE run_id = %s
                       ORDER BY iteration, attempt_number""", (run_id,))
        attempts = {}
        for t in cur.fetchall():
            attempts.setdefault(t[0], []).append({
                "attempt_number": t[1], "execution_generation": t[2], "run_attempt": t[3],
                "decision_replayed": bool(t[4]), "status": t[5], "observation": _j(t[6]),
                "error": t[7], "step_id": t[8],
                "started_at": t[9].isoformat() if t[9] else None,
                "finished_at": t[10].isoformat() if t[10] else None})
    out = []
    for r in rows:
        out.append({"iteration": r[1], "action": r[2], "arguments": _j(r[3]), "reason": r[4],
                    "decided_by": r[5], "status": r[6], "observation": _j(r[7]), "error": r[8],
                    "created_at": r[9].isoformat() if r[9] else None,
                    "finished_at": r[10].isoformat() if r[10] else None,
                    "decided_in_generation": r[11], "attempt_count": int(r[12] or 0),
                    "attempts": attempts.get(r[0], [])})
    return out


# ------------------------------------------------------------------- searches --

def search_history(run_id, provider, query_norm):
    """Every recorded execution of (provider, query) in this run, newest generation
    first. One row per execution generation (migration 0011)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT provider_status, new_jobs, duplicates, eligible_jobs,
                              rejection_summary, iteration, execution_generation,
                              provider_detail, fetched, reused_from_generation
                       FROM agent_searches
                       WHERE run_id = %s AND provider = %s AND query_norm = %s
                       ORDER BY execution_generation DESC, id DESC""",
                    (run_id, provider, query_norm))
        rows = cur.fetchall()
    return [{"provider_status": r[0], "new_jobs": r[1], "duplicates": r[2],
             "eligible_jobs": r[3], "rejection_summary": r[4], "iteration": r[5],
             "execution_generation": r[6], "provider_detail": r[7], "fetched": bool(r[8]),
             "reused_from_generation": r[9]} for r in rows]


def record_search(run_id, generation, iteration, provider, query_norm, location, obs,
                  fetched=True, reused_from_generation=None):
    """One row per (run, provider, query, execution generation). `fetched` says
    whether the provider was actually called in this generation; a reused result
    names the generation it came from."""
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation)
        cur.execute("""
            INSERT INTO agent_searches (run_id, provider, query_norm, location, iteration,
                provider_status, new_jobs, duplicates, eligible_jobs, rejection_summary,
                execution_generation, provider_detail, fetched, reused_from_generation)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, provider, query_norm, execution_generation) DO NOTHING
        """, (run_id, provider, query_norm, location or None, iteration,
              obs.get("provider_status"), obs.get("new_jobs", 0), obs.get("duplicates", 0),
              obs.get("eligible_jobs", 0), json.dumps(obs.get("rejection_summary") or {}),
              generation, obs.get("provider_detail"), bool(fetched), reused_from_generation))


# ------------------------------------------------------------------- postings --

def load_postings(job_ids):
    """Full posting rows (including description) for evaluation. Descriptions are
    NOT kept in the checkpointed state (R14) — they are loaded on demand."""
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


# -------------------------------------------------------------------- progress --

_UNVERIFIED_SQL = """(retrieved_context->>'requirements_method' = 'rule_based'
                      AND COALESCE(judge_status, '') <> 'ran'
                      AND review_status IS NULL)"""


def qualified_job_ids(run_id, qualifying_decisions, verified_only=False):
    """BACKEND-VERIFIED progress: distinct jobs whose persisted AUTHORITATIVE final
    decision qualifies and which are not still awaiting a human review. With
    verified_only, matches that rest ONLY on rule-based requirement extraction (no
    judge, no human) are excluded. Raises on database errors — an unknown count is
    never reported as zero (N07)."""
    extra = f" AND NOT {_UNVERIFIED_SQL}" if verified_only else ""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT DISTINCT (retrieved_context->>'job_id')::int
            FROM steps
            WHERE run_id = %s
              AND retrieved_context ? 'job_id'
              AND retrieved_context->>'job_id' ~ '^[0-9]+$'
              AND final_decision = ANY(%s)
              AND status = 'success'
              AND NOT (COALESCE(needs_human_review, FALSE) AND review_status IS NULL)
              {extra}
        """, (run_id, list(qualifying_decisions)))
        return sorted(r[0] for r in cur.fetchall())


def unverified_qualified_count(run_id, qualifying_decisions):
    """How many qualifying matches rest only on rule-based requirement extraction."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT COUNT(DISTINCT retrieved_context->>'job_id')
            FROM steps
            WHERE run_id = %s AND retrieved_context ? 'job_id'
              AND final_decision = ANY(%s) AND status = 'success'
              AND NOT (COALESCE(needs_human_review, FALSE) AND review_status IS NULL)
              AND {_UNVERIFIED_SQL}
        """, (run_id, list(qualifying_decisions)))
        return int(cur.fetchone()[0] or 0)


def advised_job_ids(run_id):
    """Jobs that already have persisted advice in this run (durable idempotency for
    generate_advice across checkpoints and crashes — N10)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT job_id FROM run_advice WHERE run_id = %s AND job_id IS NOT NULL",
                    (run_id,))
        return sorted(r[0] for r in cur.fetchall())


# --------------------------------------------------------------------- reviews --

def create_review_request(run_id, generation, review_id, kind, payload, step_id=None, job_id=None):
    """Create (idempotently) the immutable review identity for an interrupt. Any
    OTHER open request for the run is superseded first (one open request per run)."""
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
    """Exactly-once consumption. Returns True if THIS call consumed it."""
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation)
        cur.execute("""UPDATE review_requests SET status = 'consumed', consumed_at = NOW()
                       WHERE review_id = %s AND status = 'submitted' RETURNING review_id""",
                    (review_id,))
        return cur.fetchone() is not None


# ---------------------------------------------------------------------- output --

def persist_rankings(run_id, generation, ranked):
    """Replace the run's ranking in ONE fenced transaction; raises on failure (R03)."""
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
    """Advice summary (run_advice) + structured suggestions, bound to (run, job,
    resume version), in ONE fenced transaction. Idempotent replace (R03, R20)."""
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


FINAL_COST_TOTALS_SQL = """
    total_tokens = (SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0)
                    FROM llm_calls WHERE run_id = %(run)s),
    total_cost = (SELECT SUM(cost_usd) FROM llm_calls
                  WHERE run_id = %(run)s AND cost_status = 'priced'),
    unknown_cost_calls = (SELECT COUNT(*) FROM llm_calls
                          WHERE run_id = %(run)s AND cost_status = 'unknown')
                       + (SELECT COUNT(*) FROM llm_cost_reservations
                          WHERE run_id = %(run)s AND status = 'abandoned')"""


def finalize(run_id, generation, status, stop_reason, error_code, progress):
    """Terminal transition, fenced. Closes the active runtime interval. Any
    reservation still open is abandoned (it stays counted). Cost totals keep
    unknown cost visible: total_cost is the sum of PRICED calls (NULL if none) and
    unknown_cost_calls says how many calls are missing from it — a partial total
    is never presented as complete."""
    code = error_code.value if hasattr(error_code, "value") else error_code
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation, lock="FOR UPDATE")
        _abandon_open_reservations(cur, run_id)
        cur.execute(f"""
            UPDATE runs SET ended_at = NOW(), status = %(status)s, stop_reason = %(reason)s,
                error_code = %(code)s, pending_review = NULL, goal_progress = %(progress)s,
                {FINAL_COST_TOTALS_SQL},
                {CLOSE_EXECUTION_INTERVAL_SQL}
            WHERE id = %(run)s
        """, {"status": status, "reason": redact_secrets(stop_reason, 500), "code": code,
              "progress": json.dumps(progress, default=str), "run": run_id})
        cur.execute("""UPDATE review_requests SET status = 'superseded'
                       WHERE run_id = %s AND status IN ('pending', 'submitted')""", (run_id,))


def mark_waiting(run_id, generation, payload):
    """Pause for a human. Closes the active interval: waiting time is not runtime."""
    with get_connection() as conn:
        cur = conn.cursor()
        _check_generation(cur, run_id, generation, lock="FOR UPDATE")
        cur.execute(f"""UPDATE runs SET status = 'waiting_for_human', pending_review = %s,
                            {CLOSE_EXECUTION_INTERVAL_SQL}
                        WHERE id = %s""",
                    (json.dumps(payload, default=str), run_id))