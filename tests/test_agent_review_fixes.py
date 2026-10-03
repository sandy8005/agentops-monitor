"""Regression tests for the second review (N01-N10). Pure logic plus the REAL
LangGraph agent graph on MemorySaver with the database layer faked in memory."""
import contextlib

import pytest
from langgraph.checkpoint.memory import MemorySaver

import agent_loop
import agent_store
from agent_goal import AgentGoal
from agent_tools import ToolContext, ToolRejected, execute
from resume_advisor import rule_suggestions, validate_rewrite
from rule_resume_parser import parse_resume_rules
from sanitize import redact_secrets
from scorer import calculate_match_score
from skills import skill_mention_states


# ------------------------------------------------------------------ N02 --------

def test_rewrite_adding_deployment_is_not_validated():
    r = "Built a FastAPI service."
    st, _ = validate_rewrite("Built a FastAPI service.",
                             "Built and deployed a FastAPI service to production.", r)
    assert st != "validated"


def test_rewrite_reversing_negation_is_rejected():
    r = "Never built a FastAPI service."
    st, notes = validate_rewrite("Never built a FastAPI service.", "Built a FastAPI service.", r)
    assert st == "rejected" and "negative" in notes


def test_project_tech_must_come_from_that_projects_text():
    resume = "Projects\nPortfolio: HTML page\nAutomation: Python scripts for reports\n"
    pr = {"projects": [{"name": "Portfolio", "tech": ["Python"]},
                       {"name": "Automation", "tech": ["Python"]}],
          "years_experience": 2, "experience": [], "education": [], "skills": []}
    reqs = {"required_skills": ["Python"], "preferred_skills": [], "min_years_experience": 0,
            "responsibilities": []}
    sc = calculate_match_score(pr, reqs, resume)
    out = [s for s in rule_suggestions(pr, resume, {"title": "x"}, reqs, sc)
           if s["kind"] == "reorder_project"]
    names = [s["suggested_text"] for s in out]
    assert any("Automation" in n for n in names)
    assert not any("Portfolio" in n for n in names)


# ------------------------------------------------------------------ N05 --------

@pytest.mark.parametrize("text", [
    "Experienced Java engineer. Python experience: none.",
    "Python: none", "Python (none)", "No Python experience", "Python - 0 years",
    "Currently learning Python"])
def test_non_affirmative_python_never_matches(text):
    pr = {"skills": [], "years_experience": 5, "education": [], "projects": [],
          "experience": [{"title": "Eng", "company": "X", "years": 5}]}
    r = calculate_match_score(pr, {"required_skills": ["Python"], "preferred_skills": [],
                                   "min_years_experience": 2, "responsibilities": []}, text)
    assert r["decision"] != "Apply"
    assert r["negated_skills"] or r["uncertain_skills"]


def test_mixed_positive_and_negative_counts_positive():
    st = skill_mention_states("python", "No Java. Five years of Python.")
    assert st["affirmative"] == 1


# ------------------------------------------------------------------ N01 --------

def test_rules_parser_keeps_unknown_experience_unknown():
    p = parse_resume_rules("Python developer. Built FastAPI services.")
    assert p["years_experience"] is None and p["experience_unknown"]
    r = calculate_match_score(p, {"required_skills": ["Python"], "preferred_skills": [],
                                  "min_years_experience": 3, "responsibilities": []},
                              "Python developer. Built FastAPI services.")
    assert r["experience_unknown"] and r["decision"] != "Apply"


def test_rules_only_goal_disables_model_flags():
    g = AgentGoal(target_role="AI Engineer", model_policy="rules_only",
                  use_llm_controller=True, use_llm_advice=True, evaluate_quality=True)
    assert not g.use_llm_controller and not g.use_llm_advice and not g.evaluate_quality


def test_no_api_key_means_no_model_call(monkeypatch):
    import llm
    monkeypatch.setattr(llm.settings, "gemini_api_key", None, raising=False)
    called = []
    monkeypatch.setattr(llm, "real_llm_once", lambda p: called.append(p))
    with pytest.raises(llm.ModelNotConfigured):
        llm.logged_llm_call("x", 1, 1)
    assert not called and not llm.llm_available()


def test_quota_breaker_blocks_further_calls(monkeypatch):
    import llm
    monkeypatch.setattr(llm.settings, "gemini_api_key", "k", raising=False)
    llm.reset_quota_breaker()
    calls = []

    def boom(p):
        calls.append(p)
        raise RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    monkeypatch.setattr(llm, "real_llm_once", boom)
    monkeypatch.setattr(llm, "_log_llm_attempt", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        llm.logged_llm_call("x", 1, 1)
    with pytest.raises(llm.QuotaCircuitOpen):
        llm.logged_llm_call("x", 1, 1)
    assert len(calls) == 1 and not llm.llm_available()
    llm.reset_quota_breaker()


# ------------------------------------------------------------------ N09 --------

def test_bearer_token_fully_redacted():
    for t in ["Authorization: Bearer synthetic-access-token",
              '{"Authorization": "Bearer synthetic-access-token"}',
              "authorization=Basic synthetic-access-token"]:
        assert "synthetic-access-token" not in redact_secrets(t)


def test_timeline_redacts_nested_free_text(monkeypatch):
    import api
    monkeypatch.setattr(api, "REDACT_TRACE_PAYLOADS", True)
    a = {"action": "evaluate_jobs", "decided_by": "llm", "reason": "private reason",
         "arguments": {"job_ids": [1]}, "error": "private",
         "observation": {"failed": "PRIVATE-MARKER", "decisions": {"Apply": 1}}}
    out = api._safe_action(a)
    blob = str(out)
    assert "PRIVATE-MARKER" not in blob and "private reason" not in blob
    assert out["observation"]["decisions"] == {"Apply": 1}


# ------------------------------------------------------------ fake database ----

class FakeStore:
    def __init__(self, goal):
        self.goal = goal.model_dump()
        self.gen = 0
        self.cancel = False
        self.actions, self.searches, self.reviews = {}, {}, {}
        self.qualified = []
        self.advised = []
        self.final = None
        self.waiting = None
        self.status = "queued"

    def install(self, mp):
        s = self
        mp.setattr(agent_store, "begin_execution", lambda rid, new_attempt: s._begin())
        mp.setattr(agent_store, "load_run_config", lambda rid: {
            "goal": s.goal, "resume_id": 1, "user_id": 1,
            "cancel_requested": s.cancel, "status": s.status})
        mp.setattr(agent_store, "is_cancel_requested", lambda rid: s.cancel)
        mp.setattr(agent_store, "run_usage", lambda rid: {
            "elapsed_seconds": 1, "active_runtime_seconds": 1, "llm_calls_reserved": 0,
            "llm_call_budget": 0, "known_cost_usd": 0.0, "unknown_cost_calls": 0,
            "unknown_cost_bound_usd": 0.0, "unbounded_unknown": 0, "reserved_open_usd": 0.0,
            "committed_usd": 0.0})
        mp.setattr(agent_store, "begin_action_attempt", lambda rid, g, i, replayed=False: 1)
        mp.setattr(agent_store, "reserve_llm_call", lambda *a, **k: None)
        # Providers answer "no postings" unless a test says otherwise (no network).
        import adzuna_jobs
        import live_jobs
        mp.setattr(adzuna_jobs, "fetch_and_upsert_adzuna", lambda *a, **k: (0, 0, "empty"))
        mp.setattr(live_jobs, "fetch_and_upsert_remotive", lambda *a, **k: (0, 0, "empty"))
        mp.setattr(agent_store, "search_history", lambda *a: [])
        mp.setattr(agent_store, "record_search", lambda *a, **k: None)
        s.attempts = []
        mp.setattr(agent_store, "search_attempts", lambda rid, p, q: [
            a for a in s.attempts if (a["provider"], a["query_norm"]) == (p, q)])

        def _begin_attempt(rid, g, i, p, q):
            s.attempts.append({"id": len(s.attempts) + 1, "provider": p, "query_norm": q,
                               "execution_generation": g, "status": "started"})
            return len(s.attempts)
        mp.setattr(agent_store, "begin_search_attempt", _begin_attempt)
        mp.setattr(agent_store, "finish_search_attempt",
                   lambda rid, g, aid, st, detail=None: s.attempts[aid - 1].update(status=st))
        import llm
        mp.setattr(llm, "create_step", lambda *a: 1)
        mp.setattr(llm, "finish_step", lambda *a, **k: None)
        mp.setattr(llm, "fail_step", lambda *a, **k: None)
        mp.setattr(agent_store, "set_controller_mode", lambda *a: None)
        mp.setattr(agent_store, "get_action", lambda rid, i: s.actions.get(i))
        mp.setattr(agent_store, "record_proposed_action",
                   lambda rid, g, i, a, args, r, d: s.actions.setdefault(i, {
                       "action": a, "arguments": args, "reason": r, "decided_by": d,
                       "status": "proposed", "observation": None, "error": None}))
        mp.setattr(agent_store, "record_action_outcome",
                   lambda rid, g, i, st, obs=None, error=None, step_id=None, replayed=False:
                   s.actions[i].update(status=st, observation=obs))
        mp.setattr(agent_store, "qualified_job_ids", lambda rid, d, verified_only=False: list(s.qualified))
        mp.setattr(agent_store, "unverified_qualified_count", lambda rid, d: 0)
        mp.setattr(agent_store, "advised_job_ids", lambda rid: list(s.advised))
        mp.setattr(agent_store, "create_review_request",
                   lambda rid, g, review_id, kind, payload, step_id=None, job_id=None:
                   s.reviews.setdefault(review_id, {"status": "pending", "answer": None,
                                                    "decision": None}))
        mp.setattr(agent_store, "review_status", lambda review_id: s.reviews.get(review_id))
        mp.setattr(agent_store, "mark_review_consumed",
                   lambda review_id, g, rid: s.reviews[review_id].update(status="consumed"))
        mp.setattr(agent_store, "persist_rankings", lambda rid, g, ranked: None)
        mp.setattr(agent_store, "finalize", lambda rid, g, st, reason, code, prog: s._final(st, reason))
        mp.setattr(agent_store, "mark_waiting", lambda rid, g, payload: s._wait(payload))

    def _begin(self):
        self.gen += 1
        self.status = "running"
        return self.gen

    def _final(self, st, reason):
        self.final = (st, reason)
        self.status = st

    def _wait(self, payload):
        self.waiting = payload
        self.status = "waiting_for_human"


@pytest.fixture
def env(monkeypatch):
    goal = AgentGoal(target_role="AI Engineer", target_count=1, model_policy="rules_only",
                     providers=["adzuna"], limits={"max_iterations": 6, "max_searches": 1})
    fs = FakeStore(goal)
    fs.install(monkeypatch)
    saver = MemorySaver()
    factory = lambda: contextlib.nullcontext(saver)
    return fs, factory, monkeypatch


def _setup(monkeypatch, fail_first=False):
    import router
    state = {"n": 0}

    def load(s, rid):
        s.resume_text = "Python developer with 3 years of experience."

    def parse(s, rid):
        state["n"] += 1
        if fail_first and state["n"] == 1:
            s.error = "parse failed: transient"
            return
        s.parsed_resume = parse_resume_rules(s.resume_text)
    monkeypatch.setattr(router, "load_resume", load)
    monkeypatch.setattr(router, "do_parse_resume", parse)


def test_N03_setup_retry_does_not_reuse_old_failure(env, monkeypatch):
    fs, factory, mp = env
    _setup(mp, fail_first=True)
    import agent_tools
    mp.setattr(agent_tools, "_collect_candidates", lambda ctx, p, q: [])
    import llm
    mp.setattr(llm, "create_step", lambda *a: 1)
    mp.setattr(llm, "finish_step", lambda *a, **k: None)
    mp.setattr(llm, "fail_step", lambda *a, **k: None)
    mp.setattr(agent_store, "search_history", lambda *a: [])
    mp.setattr(agent_store, "record_search", lambda *a, **k: None)
    agent_loop.run_agent_loop(510, checkpointer_factory=factory)
    assert fs.final[0] == "failed"
    agent_loop.run_agent_loop(510, queue_attempt=2, checkpointer_factory=factory)
    assert fs.final[0] != "failed" or "transient" not in fs.final[1]


def test_N04_completed_checkpoint_repairs_status(env, monkeypatch):
    fs, factory, mp = env
    _setup(mp)
    fs.qualified = [7]                          # goal already met -> rank, finish
    real_finalize = agent_store.finalize
    crashed = {"once": True}

    def crash_then_ok(*a):
        if crashed["once"]:
            crashed["once"] = False
            raise KeyboardInterrupt("process died before the run row was finalized")
        return real_finalize(*a)
    mp.setattr(agent_store, "finalize", crash_then_ok)
    with pytest.raises(KeyboardInterrupt):
        agent_loop.run_agent_loop(511, checkpointer_factory=factory)
    assert fs.final is None
    out = agent_loop.run_agent_loop(511, queue_attempt=2, checkpointer_factory=factory)
    assert out.get("repaired") and fs.final[0] == "success"


def test_N06_cancel_seen_never_pauses_for_review(env, monkeypatch):
    fs, factory, mp = env
    s = {"cancel_seen": True, "stop": None,
         "review_queue": [{"review_id": "1:job:5", "step_id": 5, "job_id": 5}]}
    assert agent_loop.route_after_act(s) == "finalize"
    fs.cancel = True
    out = agent_loop.node_review({"run_id": 1, "goal": fs.goal, "review_queue":
                                  s["review_queue"]},
                                 {"configurable": {"generation": 1}})
    assert out["stop"]["cancel"]


def test_N07_all_evaluations_failed_is_not_no_matches(env, monkeypatch):
    fs, factory, mp = env
    state = {"run_id": 1, "goal": fs.goal, "stop": {"by": "controller", "reason": "done"},
             "evaluated": {"5": {"job_id": 5, "status": "failed"}}, "searches": [
                 {"provider": "adzuna", "provider_status": "success"}]}
    out = agent_loop.node_finalize(state, {"configurable": {"generation": 1}})
    assert out["final"]["status"] == "failed"


def test_N07_count_error_is_not_zero(env, monkeypatch):
    fs, factory, mp = env

    def boom(*a, **k):
        raise RuntimeError("db down")
    mp.setattr(agent_store, "qualified_job_ids", boom)
    with pytest.raises(RuntimeError):
        agent_loop.node_finalize({"run_id": 1, "goal": fs.goal, "evaluated": {}},
                                 {"configurable": {"generation": 1}})


def test_N08_mode_lookup_failure_does_not_pick_pipeline(monkeypatch):
    import worker

    class Boom:
        def __enter__(self):
            raise RuntimeError("connection reset")

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(worker, "get_connection", lambda: Boom())
    with pytest.raises(worker.DatabaseUnavailable):
        worker._run_mode(5)
    from error_codes import classify_exception, ErrorCode
    assert classify_exception(worker.DatabaseUnavailable("database_unavailable: x")) \
        == ErrorCode.DATABASE_UNAVAILABLE


def test_N10_advice_repeats_and_cap_rejected(env, monkeypatch):
    fs, factory, mp = env
    fs.advised = [1, 2, 3]
    g = AgentGoal(target_role="AI Engineer")
    state = {"evaluated": {str(i): {"job_id": i, "status": "ok", "final_decision": "Apply"}
                           for i in range(1, 6)}, "advice_attempted": [1, 2, 3]}
    ctx = ToolContext(1, 1, g, state, None, 9)
    with pytest.raises(ToolRejected):
        execute(ctx, "generate_advice", {"job_ids": [1, 4]})      # repeat
    with pytest.raises(ToolRejected):
        execute(ctx, "generate_advice", {"job_ids": [4]})         # 4th job over the cap
    with pytest.raises(ToolRejected):
        execute(ctx, "generate_advice", {"job_ids": [4, 4]})      # duplicate ids