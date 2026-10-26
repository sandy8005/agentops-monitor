"""
Live smoke test: ONE tiny, cost-capped agent run against the REAL Gemini API and
the real Remotive feed, through the real worker dispatch, on the configured
database. It's an operator check before or after a deploy, not a unit test: it lives
outside tests/ (pytest's testpaths), so CI never collects it and never needs a key.

    python scripts/smoke_live.py                    # max_cost_usd 0.05 (the ceiling)
    python scripts/smoke_live.py --max-cost 0.02 --max-llm-calls 4

Needs GEMINI_API_KEY, a migrated database (python migrate.py) and outbound network.
Without a key or a database it exits 2 with a one-line reason, before any request.

What it asserts (exit 1 on any failure, 0 when all hold; check_run):
  * the outcome is success / partial_success, or no_matches ONLY when the search
    space was genuinely exhausted (error_code no_matches) with zero provider
    errors. completed_with_errors, failed, cancelled, a limit-stopped no_matches
    or any run error_code fail;
  * the model was actually called (at least one llm_calls row);
  * no failed LLM call, tool (provider) call, step or agent action attempt that was
    not recovered by a later success, and no error-coded agent action;
  * no judge returned invalid_output;
  * no rules fallback (rules-decided action, rule-based requirements, judge
    unavailable) while Gemini was expected, i.e. before the run's LLM call budget
    was spent;
  * no reservation is left open;
  * committed spend (known + unknown-cost bounds) is within the cap;
  * cost_ledger_violations(run_id) is empty.
Each check prints one metadata-only line (name, statuses, counts).

It prints metadata only (ids, counts, statuses, USD totals), never resume or
posting text. The run, its synthetic user ("smoke_<hex>") and its synthetic resume
are left in the database for inspection; erase them like any other run.
"""
import argparse
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MAX_COST_CEILING = 0.05
TERMINAL_OK = ("success", "partial_success")   # no_matches only when exhausted (check_run)
RESUME = ("Sam Synthetic. Backend engineer. 4 years of experience building services in "
          "Python, Django and PostgreSQL on AWS. Built a payments API in Python and Django.")


def _fail_setup(msg):
    print(f"smoke_live: NOT RUN (setup): {msg}", file=sys.stderr)
    return 2


def collect_run_record(run_id):
    """Metadata-only facts about one run for check_run (counts, statuses, USD)."""
    import agent_store
    from database import get_connection
    queries = {
        "llm_calls": "SELECT count(*) FROM llm_calls WHERE run_id = %(r)s",
        # a failed attempt is recovered only by a later success of the SAME logical call
        "llm_failed_unrecovered": """
            SELECT count(*) FROM llm_calls f WHERE f.run_id = %(r)s AND f.status <> 'success'
              AND NOT EXISTS (SELECT 1 FROM llm_calls s WHERE s.run_id = f.run_id
                              AND s.logical_call_id = f.logical_call_id
                              AND s.status = 'success' AND s.id > f.id)""",
        "tool_failed_unrecovered": """
            SELECT count(*) FROM tool_calls f WHERE f.run_id = %(r)s
              AND f.status IS NOT NULL AND f.status <> 'success'
              AND NOT EXISTS (SELECT 1 FROM tool_calls s WHERE s.run_id = f.run_id
                              AND s.tool_name = f.tool_name
                              AND s.input_json::text = f.input_json::text
                              AND s.status = 'success' AND s.id > f.id)""",
        "failed_steps_unrecovered": """
            SELECT count(*) FROM steps f WHERE f.run_id = %(r)s AND f.status = 'failed'
              AND NOT (f.retrieved_context ? 'job_id' AND EXISTS (
                  SELECT 1 FROM steps s WHERE s.run_id = f.run_id
                    AND s.step_name = f.step_name AND s.status = 'success' AND s.id > f.id
                    AND s.retrieved_context->>'job_id' = f.retrieved_context->>'job_id'))""",
        "action_errors": """
            SELECT (SELECT count(*) FROM agent_actions WHERE run_id = %(r)s AND error IS NOT NULL)
                 + (SELECT count(*) FROM agent_action_attempts t
                    JOIN agent_actions a ON a.id = t.action_id
                    WHERE t.run_id = %(r)s AND t.status = 'failed' AND a.status <> 'executed')""",
        "judge_invalid_output":
            "SELECT count(*) FROM steps WHERE run_id = %(r)s AND judge_status = 'invalid_output'",
        "rules_fallbacks": """
            SELECT (SELECT count(*) FROM agent_actions WHERE run_id = %(r)s AND decided_by = 'rules')
                 + (SELECT count(*) FROM steps WHERE run_id = %(r)s
                    AND (retrieved_context->>'requirements_method' = 'rule_based'
                         OR judge_status = 'unavailable'))""",
        "provider_errors": """
            SELECT (SELECT count(*) FROM agent_searches WHERE run_id = %(r)s
                    AND provider_status = 'failed')
                 + (SELECT count(*) FROM external_search_attempts WHERE run_id = %(r)s
                    AND status <> 'succeeded')""",
        "violations": "SELECT count(*) FROM cost_ledger_violations(%(r)s)",
    }
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT status, error_code, (goal_progress->>'search_exhausted')::boolean,
                              llm_call_budget, llm_calls_reserved
                       FROM runs WHERE id = %s""", (run_id,))
        status, code, exhausted, budget, reserved = cur.fetchone()
        rec = {"status": status, "error_code": code, "search_exhausted": bool(exhausted),
               "llm_budget_spent": budget is not None and (reserved or 0) >= budget}
        for name, sql in queries.items():
            cur.execute(sql, {"r": run_id})
            rec[name] = int(cur.fetchone()[0] or 0)
    usage = agent_store.run_usage(run_id)
    rec.update(open_reservations=int(usage["open_reservations"] or 0),
               committed_usd=float(usage["committed_usd"] or 0),
               known_cost_usd=float(usage["known_cost_usd"] or 0),
               unknown_cost_calls=int(usage["unknown_cost_calls"] or 0))
    return rec


def check_run(rec, max_cost):
    """[(ok, what)] for one run record (collect_run_record). Pure: no I/O, so the
    pass/fail rules are unit-tested with fake records (tests/test_smoke_live_script.py).
    `what` is metadata only (names, statuses, counts, USD). A missing count fails
    closed (treated as unknown, not as zero)."""
    def n(key):
        return rec.get(key)

    def zero(key):
        return n(key) == 0

    status, code = n("status"), n("error_code")
    exhausted_no_matches = (status == "no_matches" and code == "no_matches"
                            and n("search_exhausted") is True and zero("provider_errors"))
    outcome_ok = (status in TERMINAL_OK and code is None) or exhausted_no_matches
    committed = n("committed_usd")
    return [
        (outcome_ok, f"outcome is success/partial_success or an exhausted no_matches "
                     f"(status={status} error_code={code} exhausted={n('search_exhausted')} "
                     f"provider_errors={n('provider_errors')})"),
        (isinstance(n("llm_calls"), int) and n("llm_calls") > 0,
         f"the model was called at least once ({n('llm_calls')})"),
        (zero("llm_failed_unrecovered"),
         f"no unrecovered failed LLM call ({n('llm_failed_unrecovered')})"),
        (zero("tool_failed_unrecovered"),
         f"no unrecovered failed tool call ({n('tool_failed_unrecovered')})"),
        (zero("failed_steps_unrecovered"),
         f"no unrecovered failed step ({n('failed_steps_unrecovered')})"),
        (zero("action_errors"), f"no error-coded agent action ({n('action_errors')})"),
        (zero("judge_invalid_output"), f"no judge invalid_output ({n('judge_invalid_output')})"),
        (zero("rules_fallbacks") or n("llm_budget_spent") is True,
         f"no rules fallback while Gemini was expected ({n('rules_fallbacks')}, "
         f"llm budget spent={n('llm_budget_spent')})"),
        (zero("open_reservations"), f"no open cost reservation ({n('open_reservations')})"),
        (isinstance(committed, (int, float)) and committed <= max_cost + 1e-9,
         f"committed spend ${committed} within the ${max_cost} cap"),
        (zero("violations"), f"cost_ledger_violations empty ({n('violations')} found)"),
    ]


def report(rec, max_cost):
    """Print every check (one metadata-only line each); return the failure count."""
    results = check_run(rec, max_cost)
    for ok, what in results:
        print(f"  [{'ok' if ok else 'FAIL'}] {what}")
    return sum(1 for ok, _ in results if not ok)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-cost", type=float, default=MAX_COST_CEILING,
                    help=f"hard USD cap for the run (<= {MAX_COST_CEILING})")
    ap.add_argument("--max-llm-calls", type=int, default=6)
    ap.add_argument("--role", default="Python Developer")
    args = ap.parse_args(argv)
    if not 0 < args.max_cost <= MAX_COST_CEILING:
        return _fail_setup(f"--max-cost must be in (0, {MAX_COST_CEILING}]")
    if args.max_llm_calls < 1:
        return _fail_setup("--max-llm-calls must be >= 1")

    # 1. A real key, BEFORE touching the database or the network.
    try:
        from settings import settings
    except Exception as e:                      # invalid .env: say which kind, not the value
        return _fail_setup(f"settings could not be loaded ({type(e).__name__})")
    key = (settings.gemini_api_key or "").strip()
    if not key or key == "test-key-not-real":
        return _fail_setup("GEMINI_API_KEY is not set; the live smoke test needs a real key")

    # 2. A reachable, migrated database.
    try:
        from database import get_connection
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT to_regprocedure('cost_ledger_violations(integer)') IS NOT NULL")
            if not cur.fetchone()[0]:
                return _fail_setup("database is not migrated (run: python migrate.py)")
    except Exception as e:
        return _fail_setup(f"database unreachable ({type(e).__name__})")

    from agent_goal import AgentGoal
    from auth import create_user
    from checkpointing import setup_schema
    from job_queue import enqueue_tx
    from llm import create_run_tx
    import worker

    setup_schema()
    goal = AgentGoal(target_role=args.role, target_count=1, providers=["remotive"],
                     model_policy="auto",
                     limits={"max_iterations": 8, "max_searches": 1,
                             "max_llm_calls": args.max_llm_calls,
                             "max_cost_usd": args.max_cost})
    uid = create_user("smoke_" + uuid.uuid4().hex[:10], "pw-" + uuid.uuid4().hex)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO resumes (name, resume_text, created_at, user_id) "
                    "VALUES ('smoke (synthetic)', %s, NOW(), %s) RETURNING id", (RESUME, uid))
        resume_id = cur.fetchone()[0]
        run_id = create_run_tx(cur, "smoke_live", resume_id=resume_id,
                               target_role=goal.target_role, user_id=uid, goal=goal)
        job_id = enqueue_tx(cur, "start_run", {"run_id": run_id, "resume_id": resume_id},
                            run_id=run_id)
        lease = str(uuid.uuid4())
        cur.execute("""UPDATE job_queue SET status = 'running', attempts = attempts + 1,
                              claimed_at = NOW(), heartbeat_at = NOW(),
                              worker_id = 'smoke_live', lease_token = %s
                       WHERE id = %s AND status = 'queued'
                       RETURNING id, kind, payload, run_id, attempts, max_attempts""",
                    (lease, job_id))
        r = cur.fetchone()
    job = {"id": r[0], "kind": r[1], "payload": r[2], "run_id": r[3], "attempts": r[4],
           "max_attempts": r[5], "lease_token": lease}
    print(f"smoke_live: run {run_id} (cap ${args.max_cost}, {args.max_llm_calls} model calls)")
    worker._process(job)

    rec = collect_run_record(run_id)
    print(f"  status={rec['status']} error_code={rec['error_code']} llm_calls={rec['llm_calls']} "
          f"known=${rec['known_cost_usd']:.6f} committed=${rec['committed_usd']:.6f} "
          f"unknown_calls={rec['unknown_cost_calls']} open={rec['open_reservations']}")
    failures = report(rec, args.max_cost)
    if failures:
        print(f"smoke_live: FAILED ({failures} check(s))", file=sys.stderr)
        return 1
    print("smoke_live: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
