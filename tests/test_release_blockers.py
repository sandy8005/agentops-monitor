"""Regression tests for the production-readiness review ("release blockers").

Real PostgreSQL (marked db):
  #1  a request that would cross the hard USD cap is refused BEFORE dispatch
  #2  missing provider usage is recorded as UNKNOWN cost (NULL), never $0
  #3  a failure after dispatch is unknown (bounded); a provider rejection is not billed
  #4  LLM reservations are fenced by execution generation
      a dead worker's reservation stays counted (abandoned), never forgiven
      a dead worker's runtime is charged to its last heartbeat + grace
      concurrent migration runs are serialized by the schema lock
      a duplicate sign-up is an application error, not a raw DB error

Pure logic:
  #5  `finish` is refused until the goal is met or the search space is exhausted
  #6  EMEA / Europe / Middle East / Africa eligibility (structured, geo.py)
  #11 a stale resume never leaves the run 'running'
      provider error specificity survives finalization; DB errors are not provider errors
      advice propagates limit/cancel stops; pricing is effective-dated
      rules-based minimum years; adversarial rewrites; auth timing; upload cap
"""
import contextlib
import threading
import time
import uuid
from datetime import datetime, timezone

import pytest

import agent_loop
import agent_store
import agent_tools
from agent_goal import AgentGoal


def _utc(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc)


# ================================================================ database ====

def _db_run(max_cost=0.5, budget=40):
    from auth import create_user
    from database import get_connection
    uid = create_user("rb_" + uuid.uuid4().hex[:10], "password-1234")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode, max_cost_usd, "
                    "llm_call_budget) VALUES ('queued', 't', %s, 'agent', %s, %s) RETURNING id",
                    (uid, max_cost, budget))
        return cur.fetchone()[0]


def _sql(query, params=()):
    from database import get_connection
    with get_connection() as c:
        cur = c.cursor()
        cur.execute(query, params)
        return cur.fetchall() if cur.description else None


def _priced_call(run_id, cost):
    _sql("INSERT INTO llm_calls (run_id, prompt_tokens, completion_tokens, cost_usd, status, "
         "cost_status) VALUES (%s, 10, 5, %s, 'success', 'priced')", (run_id, cost))


@pytest.mark.db
def test_request_that_would_cross_the_cap_is_refused_before_dispatch():
    """The reviewer's example: $0.50 cap, $0.499 spent, next request up to $0.08."""
    import llm
    rid = _db_run(max_cost=0.50)
    gen = agent_store.begin_execution(rid, new_attempt=True)
    _priced_call(rid, 0.499)
    with pytest.raises(llm.CostLimitReached):
        agent_store.reserve_llm_call(rid, gen, 40, projected_usd=0.08, max_cost_usd=0.50)
    # nothing was reserved by the refused request
    assert _sql("SELECT llm_calls_reserved FROM runs WHERE id = %s", (rid,))[0][0] == 0
    ok = agent_store.reserve_llm_call(rid, gen, 40, projected_usd=0.0005, max_cost_usd=0.50)
    assert ok and ok["amount_usd"] == pytest.approx(0.0005)
    # the open reservation counts: a second one that fits only without it is refused
    with pytest.raises(llm.CostLimitReached):
        agent_store.reserve_llm_call(rid, gen, 40, projected_usd=0.0006, max_cost_usd=0.50)
    assert agent_store.run_usage(rid)["committed_usd"] == pytest.approx(0.4995)


@pytest.mark.db
def test_reservation_is_fenced_by_execution_generation():
    rid = _db_run()
    g1 = agent_store.begin_execution(rid, new_attempt=True)
    g2 = agent_store.begin_execution(rid, new_attempt=True)       # g1 superseded
    with pytest.raises(agent_store.ExecutionLost):
        agent_store.reserve_llm_call(rid, g1, 40, projected_usd=0.001, max_cost_usd=0.5)
    assert _sql("SELECT llm_calls_reserved FROM runs WHERE id = %s", (rid,))[0][0] == 0
    assert agent_store.reserve_llm_call(rid, g2, 40, projected_usd=0.001, max_cost_usd=0.5)


@pytest.fixture
def budget_run(monkeypatch):
    """A cost-capped run, executing, with a DurableBudget of its generation."""
    import llm
    monkeypatch.setattr(llm, "get_client", lambda: pytest.fail("no real provider call"))
    rid = _db_run(max_cost=0.5)
    gen = agent_store.begin_execution(rid, new_attempt=True)
    return rid, agent_loop.DurableBudget(rid, 40, max_cost_usd=0.5, generation=gen)


def _calls(rid):
    return _sql("SELECT status, cost_status, cost_usd, prompt_tokens, completion_tokens, "
                "cost_upper_bound_usd, usage_missing, reservation_id FROM llm_calls "
                "WHERE run_id = %s ORDER BY id", (rid,))


@pytest.mark.db
def test_missing_usage_is_unknown_cost_not_zero(budget_run, monkeypatch):
    import llm
    rid, budget = budget_run
    monkeypatch.setattr(llm, "real_llm_once", lambda prompt: {
        "text": "ok", "prompt_tokens": None, "completion_tokens": None,
        "usage_missing": True, "provider_request_id": None})
    assert llm.logged_llm_call("hello", rid, None, operation="t", budget=budget) == "ok"
    [(status, cstatus, cost, pt, ct, bound, missing, res_id)] = _calls(rid)
    assert (status, cstatus, cost, pt, ct, missing) == ("success", "unknown", None, None, None, True)
    assert bound and bound > 0 and res_id is not None
    # the reservation was settled in the same transaction as the call row
    assert _sql("SELECT status FROM llm_cost_reservations WHERE id = %s", (res_id,))[0][0] == "settled"
    u = agent_store.run_usage(rid)
    assert u["unknown_cost_calls"] == 1 and u["unbounded_unknown"] == 0
    assert u["reserved_open_usd"] == 0 and u["committed_usd"] == pytest.approx(bound)


@pytest.mark.db
def test_timeout_after_dispatch_is_unknown_and_bounded(budget_run, monkeypatch):
    import llm
    rid, budget = budget_run

    def timeout(prompt):
        raise TimeoutError("request timed out")
    monkeypatch.setattr(llm, "real_llm_once", timeout)
    with pytest.raises(TimeoutError):
        llm.logged_llm_call("hello", rid, None, operation="t", budget=budget, max_retries=1)
    [(status, cstatus, cost, _pt, _ct, bound, _m, _r)] = _calls(rid)
    assert (status, cstatus, cost) == ("failed", "unknown", None) and bound > 0


@pytest.mark.db
def test_provider_rejection_is_not_billed(budget_run, monkeypatch):
    import llm
    rid, budget = budget_run

    def quota(prompt):
        raise RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProject")
    monkeypatch.setattr(llm, "real_llm_once", quota)
    with pytest.raises(RuntimeError):
        llm.logged_llm_call("hello", rid, None, operation="t", budget=budget)
    [(status, cstatus, cost, _pt, _ct, bound, _m, _r)] = _calls(rid)
    assert (status, cstatus, cost, bound) == ("failed", "not_billed", None, None)


@pytest.mark.db
def test_priced_call_counts_thinking_tokens_as_output(budget_run, monkeypatch):
    import llm

    class Usage:
        prompt_token_count, candidates_token_count, thoughts_token_count = 1000, 200, 800
        tool_use_prompt_token_count = None

    class Resp:
        text, usage_metadata, response_id = "ok", Usage(), "r1"

    class Models:
        def generate_content(self, **kw):
            assert kw["config"].max_output_tokens > 0        # every request is capped
            return Resp()

    class Client:
        models = Models()
    monkeypatch.setattr(llm, "get_client", lambda: Client())
    rid, budget = budget_run
    llm.logged_llm_call("hello", rid, None, operation="t", budget=budget)
    [(_s, cstatus, cost, pt, ct, bound, _m, _r)] = _calls(rid)
    assert (cstatus, pt, ct, bound) == ("priced", 1000, 1000, None)
    from pricing import estimate_cost
    assert float(cost) == pytest.approx(estimate_cost("gemini-3.6-flash", 1000, 1000)[0], abs=1e-6)


@pytest.mark.db
def test_dead_workers_reservation_stays_counted():
    rid = _db_run(max_cost=0.5)
    g1 = agent_store.begin_execution(rid, new_attempt=True)
    res = agent_store.reserve_llm_call(rid, g1, 40, projected_usd=0.03, max_cost_usd=0.5)
    # worker 1 dies mid-request; worker 2 takes over
    g2 = agent_store.begin_execution(rid, new_attempt=True)
    status = _sql("SELECT status FROM llm_cost_reservations WHERE id = %s", (res["reservation_id"],))
    assert status[0][0] == "abandoned"
    u = agent_store.run_usage(rid)
    assert u["committed_usd"] == pytest.approx(0.03) and u["unknown_cost_calls"] == 1
    agent_store.finalize(rid, g2, "success", "done", None, {})
    assert _sql("SELECT unknown_cost_calls FROM runs WHERE id = %s", (rid,))[0][0] == 1


@pytest.mark.db
def test_crash_is_charged_to_last_heartbeat_plus_grace():
    from settings import settings
    rid = _db_run()
    agent_store.begin_execution(rid, new_attempt=True)
    # The worker started 600s ago, heartbeated 590s ago, then died silently.
    _sql("UPDATE runs SET execution_started_at = NOW() - interval '600 seconds', "
         "execution_heartbeat_at = NOW() - interval '590 seconds' WHERE id = %s", (rid,))
    agent_store.begin_execution(rid, new_attempt=True)
    charged = _sql("SELECT active_runtime_seconds FROM runs WHERE id = %s", (rid,))[0][0]
    expected = 10 + settings.execution_heartbeat_grace_seconds
    assert expected - 2 <= charged <= expected + 2              # not 600


@pytest.mark.db
def test_heartbeat_stamps_the_run():
    import job_queue
    rid = _db_run()
    agent_store.begin_execution(rid, new_attempt=True)
    _sql("UPDATE runs SET execution_heartbeat_at = NOW() - interval '1 hour' WHERE id = %s", (rid,))
    job_queue.enqueue("start_run", {"run_id": rid}, run_id=rid)
    job = job_queue.claim_next()
    while job and job["run_id"] != rid:                          # skip other tests' jobs
        job_queue.mark_done(job["id"], job["lease_token"])
        job = job_queue.claim_next()
    assert job_queue.heartbeat(job["id"], job["lease_token"])
    age = _sql("SELECT EXTRACT(EPOCH FROM NOW() - execution_heartbeat_at) FROM runs "
               "WHERE id = %s", (rid,))[0][0]
    assert age < 5
    job_queue.mark_done(job["id"], job["lease_token"])


@pytest.mark.db
def test_schema_lock_serializes_migration_sessions():
    import psycopg2
    from migrate import schema_lock
    entered = threading.Event()
    release = threading.Event()

    def holder():
        with schema_lock():
            entered.set()
            release.wait(10)
    t = threading.Thread(target=holder)
    t.start()
    try:
        assert entered.wait(10)
        with pytest.raises(psycopg2.errors.LockNotAvailable):
            with schema_lock(timeout=1):
                pass
    finally:
        release.set()
        t.join()
    with schema_lock(timeout=5):                                  # free again
        pass


@pytest.mark.db
def test_duplicate_username_is_an_application_error():
    from auth import UsernameTaken, create_user
    name = "dup_" + uuid.uuid4().hex[:10]
    create_user(name, "password-1234")
    with pytest.raises(UsernameTaken):
        create_user(name, "password-1234")


@pytest.mark.db
def test_legacy_pipeline_job_is_closed_not_executed(monkeypatch):
    import worker
    rid = _db_run()
    # A historical legacy row: only an explicitly retired run may carry mode='pipeline'.
    _sql("UPDATE runs SET mode = 'pipeline', goal_retired_at = NOW(), "
         "goal_retired_reason = 'pipeline_engine', status = 'queued' WHERE id = %s", (rid,))
    monkeypatch.setattr(agent_loop, "run_agent_loop", lambda *a, **k: pytest.fail("executed"))
    worker._run_job({"kind": "start_run", "payload": {"run_id": rid}, "run_id": rid})
    status, code = _sql("SELECT status, error_code FROM runs WHERE id = %s", (rid,))[0]
    assert (status, code) == ("failed", "engine_retired")


# ============================================================== pure logic ====

@pytest.fixture
def no_db(monkeypatch):
    monkeypatch.setattr(agent_store, "is_cancel_requested", lambda rid: False)
    monkeypatch.setattr(agent_loop.run_lock, "check_owner", lambda rid: None)
    monkeypatch.setattr(agent_store, "qualified_job_ids", lambda *a, **k: [])
    monkeypatch.setattr(agent_store, "unverified_qualified_count", lambda *a: 0)
    monkeypatch.setattr(agent_store, "set_controller_mode", lambda *a: None)
    monkeypatch.setattr(agent_store, "begin_action_attempt", lambda *a, **k: 1)
    monkeypatch.setattr(agent_store, "run_usage", lambda rid: {
        "elapsed_seconds": 1, "active_runtime_seconds": 1, "llm_calls_reserved": 0,
        "llm_call_budget": None, "known_cost_usd": 0.0, "unknown_cost_calls": 0,
        "unknown_cost_bound_usd": 0.0, "unbounded_unknown": 0, "reserved_open_usd": 0.0,
        "committed_usd": 0.0})
    outcomes = []
    monkeypatch.setattr(agent_store, "record_action_outcome",
                        lambda rid, g, i, status, obs=None, **k: outcomes.append((status, obs)))
    return outcomes


GOAL = AgentGoal(target_role="AI Engineer", target_count=3, providers=["adzuna", "remotive"],
                 limits={"max_searches": 6})


def test_llm_cannot_finish_on_the_first_turn(no_db):
    """Reviewer's reproduction: qualified=0, nothing discovered, no search yet."""
    state = {"run_id": 1, "goal": GOAL.model_dump(), "iteration": 1, "searches": [],
             "discovered": {}, "evaluated": {},
             "pending": {"action": "finish", "arguments": {"reason": "no matches"},
                         "reason": "done", "decided_by": "llm", "replayed": False}}
    out = agent_loop.node_act(state, {"configurable": {"generation": 1}})
    assert not out.get("stop")
    assert no_db[-1][0] == "rejected"
    # rules-decided or replayed, the tool itself refuses too
    ctx = agent_tools.ToolContext(1, 1, GOAL, {"searches": [], "discovered": {}}, None, 1)
    with pytest.raises(agent_tools.ToolRejected, match="not exhausted"):
        agent_tools.execute(ctx, "finish", {"reason": "done"})


def test_finish_allowed_once_search_space_exhausted(no_db):
    from agent_goal import candidate_queries
    searches = [{"provider": p, "query": q, "provider_status": "success"}
                for q in candidate_queries(GOAL) for p in GOAL.providers][:GOAL.limits.max_searches]
    ok, why = agent_tools.finish_permitted(GOAL, {"searches": searches, "discovered": {}}, 0)
    assert ok and "search limit" in why
    # but never while discovered eligible jobs are unevaluated
    st = {"searches": searches, "discovered": {"5": {"eligible": True}}, "evaluated": {}}
    assert not agent_tools.finish_permitted(GOAL, st, 0)[0]


def test_unusable_provider_counts_as_exhausted():
    from agent_goal import candidate_queries
    g = AgentGoal(target_role="AI Engineer", providers=["adzuna", "remotive"],
                  limits={"max_searches": 20})
    searches = [{"provider": "adzuna", "query": candidate_queries(g)[0],
                 "provider_status": "failed", "provider_detail": "auth_error"}]
    searches += [{"provider": "remotive", "query": q, "provider_status": "success"}
                 for q in candidate_queries(g)]
    space = agent_tools.search_space(g, {"searches": searches})
    assert space["exhausted"] and space["unusable_providers"] == ["adzuna"]


def _finalize(state):
    return agent_loop.node_finalize({"run_id": 1, "goal": GOAL.model_dump(), "evaluated": {},
                                     **state}, {"configurable": {"generation": 1}})["final"]


def test_run_that_never_searched_is_not_no_matches(no_db):
    f = _finalize({"stop": {"by": "limit", "reason": "runtime limit"}, "searches": []})
    assert (f["status"], f["error_code"]) == ("failed", "limit_reached")


def test_limit_before_exhaustion_is_labelled(no_db):
    f = _finalize({"stop": {"by": "limit", "reason": "iteration limit"},
                   "searches": [{"provider": "adzuna", "query": "ai engineer",
                                 "provider_status": "success"}]})
    assert (f["status"], f["error_code"]) == ("no_matches", "limit_reached")


@pytest.mark.parametrize("detail,code", [
    ("auth_error", "job_source_auth_failed"),         # terminal — not retried
    ("missing_keys", "job_source_auth_failed"),
    ("rate_limited", "job_source_rate_limited"),
    ("server_error", "job_source_unavailable"),
])
def test_all_providers_failed_keeps_specific_code(no_db, detail, code):
    f = _finalize({"stop": {"by": "controller", "reason": "done"},
                   "searches": [{"provider": "adzuna", "query": "ai engineer",
                                 "provider_status": "failed", "provider_detail": detail}]})
    assert (f["status"], f["error_code"]) == ("failed", code)


def test_database_error_during_tool_fails_the_run_not_the_action(no_db, monkeypatch):
    import psycopg2

    def boom(ctx, action, args):
        raise psycopg2.OperationalError("server closed the connection")
    monkeypatch.setattr(agent_loop, "execute", boom)
    state = {"run_id": 1, "goal": GOAL.model_dump(), "iteration": 2,
             "pending": {"action": "search_jobs", "arguments": {}, "reason": "r",
                         "decided_by": "rules", "replayed": False}}
    with pytest.raises(psycopg2.OperationalError):
        agent_loop.node_act(state, {"configurable": {"generation": 1}})


def test_provider_persistence_error_is_not_a_provider_status(monkeypatch):
    import psycopg2
    import adzuna_jobs
    job = {"external_id": "adzuna:1", "title": "t", "company": "c", "description": "d",
           "location": "", "work_mode": "", "employment_type": "", "source": "adzuna",
           "posted_at": None, "apply_url": None}
    monkeypatch.setattr(adzuna_jobs, "fetch_adzuna_jobs", lambda *a: ([job], "success", None))

    class Conn:
        def close(self):
            pass
    monkeypatch.setattr(adzuna_jobs, "_get_connection", lambda: Conn())

    def fail(*a, **k):
        raise psycopg2.OperationalError("db down")
    monkeypatch.setattr(adzuna_jobs, "upsert_adzuna_jobs", fail)
    with pytest.raises(psycopg2.OperationalError):
        adzuna_jobs.fetch_and_upsert_adzuna("x")


def test_advice_propagates_a_limit_stop(monkeypatch):
    import router
    monkeypatch.setattr(agent_store, "advised_job_ids", lambda rid: [])
    monkeypatch.setattr(agent_store, "load_postings", lambda ids: {})
    import llm
    monkeypatch.setattr(llm, "create_step", lambda *a: 9)
    steps = []
    monkeypatch.setattr(llm, "finish_step", lambda *a, **k: steps.append("ok"))
    monkeypatch.setattr(llm, "fail_step", lambda *a, **k: steps.append("failed"))
    monkeypatch.setattr(router, "resume_content_hash", lambda t: "h")
    state = {"evaluated": {"1": {"job_id": 1, "status": "ok", "final_decision": "Apply"}}}
    stop = {"by": "limit", "reason": "runtime limit reached"}
    ctx = agent_tools.ToolContext(1, 1, GOAL, state, None, 4, limit_check=lambda: stop)
    obs, _sid, _p = agent_tools.execute(ctx, "generate_advice", {"job_ids": [1]})
    assert state["limit_stop"] == stop and obs["stopped"] and obs["not_advised"] == [1]
    assert steps == ["failed"]                     # the step is not a clean success


# --------------------------------------------------------- stale resume ----

class _Snap:
    def __init__(self, nxt, values=None):
        self.next = nxt
        self.values = values or {}


def test_stale_resume_with_consumed_review_continues_the_graph(monkeypatch):
    monkeypatch.setattr(agent_store, "review_status", lambda rid: {"status": "consumed"})
    handled = []
    monkeypatch.setattr(agent_loop, "_handle_result", lambda *a: handled.append(a) or "ok")

    class G:
        def invoke(self, value, config=None):
            assert value is None                   # continue, never re-apply
            return {"final": {}}
    out = agent_loop._resume_without_interrupt(G(), {}, 1, 2, "1:job:5", _Snap(("decide",)))
    assert out == "ok" and handled


@pytest.mark.parametrize("snap", [None, _Snap(()), _Snap(("decide",))])
def test_stale_resume_otherwise_fails_with_checkpoint_error(monkeypatch, snap):
    monkeypatch.setattr(agent_store, "review_status", lambda rid: {"status": "pending"})
    finals = []
    monkeypatch.setattr(agent_store, "finalize", lambda *a: finals.append(a))
    out = agent_loop._resume_without_interrupt(object(), {}, 1, 2, "1:job:5", snap)
    assert out["status"] == "failed" and finals[0][2] == "failed"
    assert str(finals[0][4]) == "checkpoint_error"


# ----------------------------------------------------------- geography ----

@pytest.mark.parametrize("posting,candidate,expected", [
    ("Europe", "UAE", "ineligible"),
    ("Middle East", "Germany", "ineligible"),
    ("Africa", "France", "ineligible"),
    ("Europe only", "Nigeria", "ineligible"),
    ("EMEA", "Dubai, UAE", "eligible"),
    ("EMEA", "Berlin, Germany", "eligible"),
    ("EMEA", "London, UK", "eligible"),
    ("EMEA", "Nairobi, Kenya", "eligible"),
    ("Europe", "London, UK", "eligible"),
    ("UK only", "Germany", "ineligible"),
    ("Europe", "EMEA", "unknown"),              # candidate named a broader region
    ("USA only", "Canada", "ineligible"),
    ("North America", "United States", "eligible"),
    ("EMEA", "United States", "ineligible"),
])
def test_emea_hierarchy(posting, candidate, expected):
    from job_source import geo_eligibility
    assert geo_eligibility({"location": posting}, candidate) == expected


# ------------------------------------------------------------- pricing ----

def test_pricing_is_effective_dated():
    from pricing import estimate_cost, rates_for
    assert rates_for("gemini-3.6-flash", _utc(2026, 9, 29))[:2] == (0.75, 3.75)
    assert rates_for("gemini-3.6-flash", _utc(2026, 12, 31))[:2] == (0.75, 3.75)
    assert rates_for("gemini-3.6-flash", _utc(2027, 1, 1))[:2] == (1.50, 7.50)
    assert rates_for("gemini-3.6-flash", _utc(2026, 8, 1))[:2] == (1.50, 7.50)
    assert rates_for("gemini-3.6-flash", _utc(2026, 1, 1)) == (None, None, None)
    assert estimate_cost("gemini-3.6-flash", 10, None)[0] is None       # usage unknown
    jan = estimate_cost("gemini-3.6-flash", 1_000_000, 0, at=_utc(2027, 1, 2))
    assert jan == (1.5, "gemini-3.6-flash@2027-01-01")


def test_max_request_cost_is_an_upper_bound():
    from pricing import estimate_cost, max_request_cost
    prompt = "héllo wörld " * 100
    bound = max_request_cost("gemini-3.6-flash", len(prompt.encode()), 8192)
    # a tokenizer never yields more tokens than bytes; output is capped by the request
    worst = estimate_cost("gemini-3.6-flash", len(prompt.encode()), 8192)[0]
    assert bound >= worst


# ------------------------------------------------- rule-based requirements ----

@pytest.mark.parametrize("text,years", [
    ("3 years Python preferred. 5 years leadership experience for manager position.", 0.0),
    ("Requirements: 5+ years of experience in ML. 3 years Python.", 3.0),
    ("We have been growing for over 20 years. 2-4 years experience required.", 2.0),
    ("Minimum 4 years of backend experience; 7+ years is a plus.", 4.0),
    ("Founded 12 years ago. No experience requirement.", 0.0),
])
def test_rule_based_minimum_years(text, years):
    from rule_requirements import _years_from_text
    assert _years_from_text(text.lower()) == years


# ------------------------------------------------------ adversarial rewrites ----

_RESUME = ("Built a Python data pipeline at Acme Corp in 2022. Currently learning Kubernetes. "
           "No experience with Java. Worked on a team project for search ranking.")


@pytest.mark.parametrize("original,suggested,expected", [
    # invented employer
    ("Built a Python data pipeline at Acme Corp in 2022.",
     "Built a Python data pipeline at Google in 2022.", "rejected"),
    # invented certification (non-technology, so only the credential rule catches it)
    ("Worked on a team project for search ranking.",
     "Certified Scrum Master; worked on a team project for search ranking.", "rejected"),
    ("Worked on a team project for search ranking.",
     "PMP: worked on a team project for search ranking.", "rejected"),
    # invented project ownership — never auto-validated
    ("Worked on a team project for search ranking.",
     "Led and owned the search ranking project end to end.", "needs_confirmation"),
    # invented percentage improvement
    ("Built a Python data pipeline at Acme Corp in 2022.",
     "Built a Python data pipeline at Acme Corp in 2022, cutting latency by 40%.", "rejected"),
    # invented years
    ("Built a Python data pipeline at Acme Corp in 2022.",
     "Built Python data pipelines at Acme Corp for 5 years.", "rejected"),
    # negated technology turned affirmative
    ("No experience with Java.", "Experience with Java.", "rejected"),
    # "learning X" turned into "experienced in X"
    ("Currently learning Kubernetes.", "Experienced in Kubernetes.", "rejected"),
    ("Currently learning Kubernetes.", "Proficient with Kubernetes.", "rejected"),
    # a faithful rewording still validates
    ("Built a Python data pipeline at Acme Corp in 2022.",
     "Built a Python data pipeline for Acme Corp in 2022.", "validated"),
])
def test_adversarial_rewrites(original, suggested, expected):
    from resume_advisor import validate_rewrite
    status, _notes = validate_rewrite(original, suggested, _RESUME,
                                      ["python", "kubernetes", "java"])
    assert status == expected


# ------------------------------------------------------------------ auth ----

def test_unknown_username_does_equal_bcrypt_work(monkeypatch):
    import auth

    class Cur:
        def execute(self, *a):
            pass

        def fetchone(self):
            return None

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return Cur()
    monkeypatch.setattr(auth, "get_connection", lambda: Conn())
    checked = []
    monkeypatch.setattr(auth, "verify_password", lambda pw, h: checked.append(h) or False)
    assert auth.authenticate("nobody", "password-1234") is None
    assert checked and checked[0].startswith("$2")              # a real bcrypt hash


# ---------------------------------------------------------------- upload ----

def test_oversized_upload_is_rejected_while_streaming(monkeypatch):
    import asyncio
    import api

    class Big:
        def __init__(self, n):
            self.left = n
            self.reads = 0

        async def read(self, size):
            self.reads += 1
            if self.left <= 0:
                return b""
            k = min(size, self.left)
            self.left -= k
            return b"x" * k
    f = Big(10 * 1024 * 1024)
    with pytest.raises(api._UploadTooLarge):
        asyncio.run(api._read_capped(f, 256 * 1024))
    assert f.reads <= 256 // 64 + 1                            # stopped early
    assert asyncio.run(api._read_capped(Big(1000), 256 * 1024)) == b"x" * 1000


# ------------------------------------------------------ fail loudly ----

def test_worker_exits_nonzero_when_checkpoint_setup_fails(monkeypatch):
    import checkpointing
    import worker

    def broken():
        raise RuntimeError("cannot create checkpoint tables")
    monkeypatch.setattr(checkpointing, "setup_schema", broken)
    monkeypatch.setattr(worker.job_queue, "claim_next",
                        lambda: pytest.fail("a worker that cannot checkpoint must not claim jobs"))
    with pytest.raises(SystemExit) as exc:
        worker.main()
    assert exc.value.code == 2


def test_database_error_in_requirements_extraction_is_not_a_rules_fallback(monkeypatch):
    import psycopg2
    import router
    from agent_state import AgentState
    monkeypatch.setattr(router, "_reqs_cache_get", lambda h: (None, None))
    monkeypatch.setattr(router, "_llm_allowed", lambda s: True)

    def db_down(*a, **k):
        raise psycopg2.OperationalError("connection reset")
    monkeypatch.setattr(router, "extract_requirements", db_down)
    monkeypatch.setattr(router, "_reqs_cache_put", lambda *a: pytest.fail("must not cache rules"))
    with pytest.raises(psycopg2.OperationalError):
        router._get_requirements(AgentState(goal="g"), {"title": "t", "description": "d"}, 1, 1)


def test_model_outage_in_requirements_extraction_takes_the_rules_fallback(monkeypatch):
    import router
    from agent_state import AgentState
    monkeypatch.setattr(router, "_reqs_cache_get", lambda h: (None, None))
    monkeypatch.setattr(router, "_llm_allowed", lambda s: True)

    def outage(*a, **k):
        raise TimeoutError("request timed out")
    monkeypatch.setattr(router, "extract_requirements", outage)
    cached = []
    monkeypatch.setattr(router, "_reqs_cache_put", lambda h, r, m: cached.append(m))
    _reqs, hit, method = router._get_requirements(
        AgentState(goal="g"), {"title": "Python dev", "description": "Python required"}, 1, 1)
    assert (hit, method, cached) == (False, "rule_based", ["rule_based"])


def test_loop_state_keys_survive_between_graph_nodes():
    """LangGraph drops keys a node returns that the state schema does not declare.
    The finalizer depends on these being carried from `act` to `finalize`."""
    from langgraph.graph import END, StateGraph

    def act(s):
        return {"last_failure_code": "job_source_auth_failed", "search_exhausted": True}

    def finalize(s):
        return {"final": {"code": s.get("last_failure_code"),
                          "exhausted": s.get("search_exhausted")}}
    g = StateGraph(agent_loop.LoopState)
    g.add_node("act", act)
    g.add_node("finalize", finalize)
    g.set_entry_point("act")
    g.add_edge("act", "finalize")
    g.add_edge("finalize", END)
    out = g.compile().invoke({"run_id": 1})["final"]
    assert out == {"code": "job_source_auth_failed", "exhausted": True}


def test_database_error_during_advice_is_not_a_per_job_failure(monkeypatch):
    import psycopg2
    import router
    import resume_advisor
    monkeypatch.setattr(agent_store, "advised_job_ids", lambda rid: [])
    monkeypatch.setattr(agent_store, "load_postings",
                        lambda ids: {1: {"id": 1, "title": "t", "description": "d"}})
    import llm
    monkeypatch.setattr(llm, "create_step", lambda *a: 9)
    monkeypatch.setattr(llm, "finish_step", lambda *a, **k: None)
    monkeypatch.setattr(llm, "fail_step", lambda *a, **k: None)
    monkeypatch.setattr(router, "_reqs_cache_get", lambda h: (None, None))

    def db_down(*a, **k):
        raise psycopg2.OperationalError("connection reset")
    monkeypatch.setattr(resume_advisor, "build_suggestions", db_down)
    state = {"evaluated": {"1": {"job_id": 1, "status": "ok", "final_decision": "Apply"}},
             "resume_text": "Python", "parsed_resume": {"skills": ["python"]}}
    ctx = agent_tools.ToolContext(1, 1, GOAL, state, None, 4)
    with pytest.raises(psycopg2.OperationalError):
        agent_tools.execute(ctx, "generate_advice", {"job_ids": [1]})


@pytest.mark.db
def test_cancelled_legacy_run_ends_cancelled_not_engine_retired(monkeypatch):
    import worker
    rid = _db_run()
    _sql("UPDATE runs SET mode = 'pipeline', goal_retired_at = NOW(), "
         "goal_retired_reason = 'pipeline_engine', status = 'queued', cancel_requested = TRUE "
         "WHERE id = %s", (rid,))
    monkeypatch.setattr(agent_loop, "resume_agent_loop", lambda *a, **k: pytest.fail("executed"))
    worker._run_job({"kind": "resume_run", "payload": {"run_id": rid}, "run_id": rid})
    status, code = _sql("SELECT status, error_code FROM runs WHERE id = %s", (rid,))[0]
    assert (status, code) == ("cancelled", "cancelled")


@pytest.mark.parametrize("raw", ["not json at all", '{"required_skills": "python"}', "[]"])
def test_malformed_requirements_output_is_a_degraded_model_error(monkeypatch, raw):
    """Bad model JSON must take the rules fallback, never fail the job as a bug."""
    import job_parser
    import llm
    monkeypatch.setattr(job_parser, "logged_llm_call", lambda *a, **k: raw)
    with pytest.raises(llm.ModelOutputInvalid) as exc:
        job_parser.extract_requirements({"title": "t", "description": "d"}, 1, 1)
    assert llm.is_degraded_model_error(exc.value)


def test_malformed_evaluator_output_is_a_degraded_model_error(monkeypatch):
    import evaluator
    import llm
    monkeypatch.setattr(evaluator, "logged_llm_call", lambda *a, **k: "{bad")
    with pytest.raises(llm.ModelOutputInvalid) as exc:
        evaluator.evaluate_decision("resume", {"title": "t", "description": "d"}, "x", 1, 1)
    assert llm.is_degraded_model_error(exc.value)