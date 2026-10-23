"""
Live smoke test: ONE tiny, cost-capped agent run against the REAL Gemini API and
the real Remotive feed, through the real worker dispatch, on the configured
database. It's an operator check before or after a deploy, not a unit test: it lives
outside tests/ (pytest's testpaths), so CI never collects it and never needs a key.

    python scripts/smoke_live.py                    # max_cost_usd 0.05 (the ceiling)
    python scripts/smoke_live.py --max-cost 0.02 --max-llm-calls 4

Needs GEMINI_API_KEY, a migrated database (python migrate.py) and outbound network.
Without a key or a database it exits 2 with a one-line reason, before any request.

What it asserts (exit 1 on any failure, 0 when all hold):
  * the run reaches a terminal, non-failed status;
  * the model was actually called (at least one llm_calls row), and no reservation
    is left open;
  * committed spend (known + unknown-cost bounds) is within the cap;
  * cost_ledger_violations(run_id) is empty.

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
TERMINAL_OK = ("success", "partial_success", "no_matches", "completed_with_errors")
RESUME = ("Sam Synthetic. Backend engineer. 4 years of experience building services in "
          "Python, Django and PostgreSQL on AWS. Built a payments API in Python and Django.")


def _fail_setup(msg):
    print(f"smoke_live: NOT RUN (setup): {msg}", file=sys.stderr)
    return 2


def _check(ok, what, failures):
    print(f"  [{'ok' if ok else 'FAIL'}] {what}")
    if not ok:
        failures.append(what)


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

    import agent_store
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status, error_code FROM runs WHERE id = %s", (run_id,))
        status, code = cur.fetchone()
        cur.execute("SELECT count(*), count(*) FILTER (WHERE status = 'success') "
                    "FROM llm_calls WHERE run_id = %s", (run_id,))
        calls, ok_calls = cur.fetchone()
        cur.execute("SELECT problem FROM cost_ledger_violations(%s)", (run_id,))
        violations = [row[0] for row in cur.fetchall()]
    usage = agent_store.run_usage(run_id)
    print(f"  status={status} error_code={code} llm_calls={calls} (success {ok_calls}) "
          f"known=${usage['known_cost_usd']:.6f} committed=${usage['committed_usd']:.6f} "
          f"unknown_calls={usage['unknown_cost_calls']} open={usage['open_reservations']}")

    failures = []
    _check(status in TERMINAL_OK, f"terminal non-failed status ({status})", failures)
    _check(calls > 0, "the model was called at least once", failures)
    _check(usage["open_reservations"] == 0, "no open cost reservation", failures)
    _check(usage["committed_usd"] <= args.max_cost + 1e-9,
           f"committed spend within the ${args.max_cost} cap", failures)
    _check(not violations, f"cost_ledger_violations empty ({len(violations)} found)", failures)
    if failures:
        print(f"smoke_live: FAILED ({len(failures)} check(s))", file=sys.stderr)
        return 1
    print("smoke_live: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
