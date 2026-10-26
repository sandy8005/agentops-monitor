"""Round 6, backlog C: smoke_live fails on ANY unrecovered provider error (whatever
the run status), and accepts a rules fallback only with a RECORDED allowed reason
(the LLM budget was spent, the USD cap refused the request, or the model was
degraded, recorded as such). An unexplained fallback fails. The cost-cap fallback
passes (it used to fail when the request-count budget was not spent).

Every fallback now records a metadata-only code (llm.fallback_code /
router._llm_unavailable_code): agent_actions.fallback_code (migration 0020) for a
rules-decided controller action, retrieved_context.requirements_fallback_code and
judge_fallback_code for a job step. Synthetic data only."""
import importlib.util
import json
import os
import uuid

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "smoke_live.py")


def _load():
    spec = importlib.util.spec_from_file_location("smoke_live_fallbacks", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _good(**over):
    rec = {"status": "success", "error_code": None, "search_exhausted": False,
           "llm_budget_spent": False, "llm_calls": 4, "llm_failed_unrecovered": 0,
           "tool_failed_unrecovered": 0, "failed_steps_unrecovered": 0, "action_errors": 0,
           "judge_invalid_output": 0, "rules_fallbacks": 0, "rules_fallback_reasons": {},
           "provider_errors": 0, "provider_errors_unrecovered": 0,
           "open_reservations": 0, "committed_usd": 0.01, "known_cost_usd": 0.01,
           "unknown_cost_calls": 0, "violations": 0}
    rec.update(over)
    return rec


def _failed(rec):
    return [what for ok, what in _load().check_run(rec, 0.05) if not ok]


# --- provider errors --------------------------------------------------------

def test_counterexample_success_run_with_failed_search_fails():
    failed = _failed(_good(provider_errors=1, provider_errors_unrecovered=1))
    assert len(failed) == 1 and "unrecovered provider error" in failed[0], failed


@pytest.mark.parametrize("status", ["success", "partial_success"])
def test_unrecovered_provider_error_fails_whatever_the_status(status):
    failed = _failed(_good(status=status, provider_errors=2, provider_errors_unrecovered=2))
    assert any("unrecovered provider error" in w for w in failed)


def test_recovered_provider_error_passes():
    assert _failed(_good(provider_errors=1, provider_errors_unrecovered=0)) == []


def test_missing_provider_count_fails_closed():
    assert any("unrecovered provider error" in w
               for w in _failed(_good(provider_errors_unrecovered=None)))


# --- rules fallbacks --------------------------------------------------------

@pytest.mark.parametrize("reasons", [
    {"cost_limit_reached": 2},              # the cost cap refused the request: allowed
    {"llm_budget_spent": 3},
    {"llm_unavailable": 1, "llm_rate_limited": 1},
    {"llm_quota_exhausted": 1, "model_output_invalid": 1, "llm_invalid_response": 1},
    {"cost_limit_reached": 1, "llm_budget_spent": 1},
])
def test_fallback_with_an_allowed_recorded_reason_passes(reasons):
    rec = _good(rules_fallbacks=sum(reasons.values()), rules_fallback_reasons=reasons)
    assert _failed(rec) == []


def test_cost_cap_fallback_passes_without_the_call_budget_spent():
    """The earlier false fail: the USD cap stopped the model while the request
    count budget still had room."""
    rec = _good(rules_fallbacks=1, rules_fallback_reasons={"cost_limit_reached": 1},
                llm_budget_spent=False)
    assert _failed(rec) == []


@pytest.mark.parametrize("reasons", [
    {"unrecorded": 1},
    {"controller_actions_rejected": 1},
    {"controller_output_invalid": 1},
    {"llm_not_configured": 1},
    {"cost_unknown": 1},
    {"internal_error": 1},
    {"rules_only": 1},
    {"llm_budget_spent": 2, "unrecorded": 1},
])
def test_unexplained_fallback_fails(reasons):
    rec = _good(rules_fallbacks=sum(reasons.values()), rules_fallback_reasons=reasons)
    failed = _failed(rec)
    assert len(failed) == 1 and "rules fallback" in failed[0], failed


def test_budget_spent_flag_alone_does_not_excuse_an_unrecorded_fallback():
    rec = _good(rules_fallbacks=1, rules_fallback_reasons={"unrecorded": 1},
                llm_budget_spent=True)
    assert any("rules fallback" in w for w in _failed(rec))


@pytest.mark.parametrize("reasons,total", [(None, 1), ({"llm_budget_spent": 1}, 3),
                                           ({}, 2)])
def test_fallback_reasons_must_cover_every_fallback(reasons, total):
    rec = _good(rules_fallbacks=total, rules_fallback_reasons=reasons)
    assert any("rules fallback" in w for w in _failed(rec))


def test_failing_lines_are_metadata_only(capsys):
    rec = _good(rules_fallbacks=2,
                rules_fallback_reasons={"unrecorded": 1, "Senior Python Dev at Acme": 1},
                provider_errors=1, provider_errors_unrecovered=1)
    n = _load().report(rec, 0.05)
    out = capsys.readouterr().out
    assert n == 2 and out.count("[FAIL]") == 2
    assert "Acme" not in out and "Python" not in out          # odd codes are not echoed
    assert all(len(line) < 200 for line in out.splitlines())


# --- the fallback code is recorded where the fallback happens -----------------

@pytest.mark.parametrize("make,code", [
    (lambda llm: llm.CostLimitReached("cost limit: x"), "cost_limit_reached"),
    (lambda llm: llm.BudgetExceeded("LLM request budget reached (4)"), "llm_budget_spent"),
    (lambda llm: llm.CostUnknown("cost_unknown: no price"), "cost_unknown"),
    (lambda llm: llm.ModelOutputInvalid("not json"), "model_output_invalid"),
    (lambda llm: llm.QuotaCircuitOpen("quota exhausted (circuit open)"), "llm_quota_exhausted"),
    (lambda llm: llm.ModelNotConfigured("GEMINI_API_KEY is not set"), "llm_not_configured"),
    (lambda llm: TimeoutError("request timed out"), "llm_unavailable"),
])
def test_fallback_code_classifies_degraded_errors(make, code):
    import llm
    assert llm.fallback_code(make(llm)) == code


def test_requirements_cost_cap_fallback_records_its_code(monkeypatch):
    import llm
    import router
    from agent_state import AgentState
    monkeypatch.setattr(router, "_reqs_cache_get", lambda h: (None, None))
    monkeypatch.setattr(router, "_llm_allowed", lambda s: True)

    def capped(*a, **k):
        raise llm.CostLimitReached("cost limit: would exceed")
    monkeypatch.setattr(router, "extract_requirements", capped)
    monkeypatch.setattr(router, "_reqs_cache_put", lambda *a: None)
    fb = {}
    _r, _hit, method = router._get_requirements(
        AgentState(goal="g"), {"title": "Python dev", "description": "Python required"},
        1, 1, fallback=fb)
    assert method == "rule_based" and fb == {"code": "cost_limit_reached"}


def test_requirements_budget_spent_fallback_records_its_code(monkeypatch):
    import router
    from agent_state import AgentState
    monkeypatch.setattr(router, "_reqs_cache_get", lambda h: (None, None))
    monkeypatch.setattr(router, "_reqs_cache_put", lambda *a: None)
    s = AgentState(goal="g")
    monkeypatch.setattr(s, "budget_exceeded", lambda: True, raising=False)
    fb = {}
    _r, _hit, method = router._get_requirements(
        s, {"title": "Python dev", "description": "Python required"}, 1, 1, fallback=fb)
    assert method == "rule_based" and fb == {"code": "llm_budget_spent"}


def test_requirements_llm_path_records_no_fallback(monkeypatch):
    import router
    from agent_state import AgentState
    monkeypatch.setattr(router, "_reqs_cache_get", lambda h: (None, None))
    monkeypatch.setattr(router, "_llm_allowed", lambda s: True)
    monkeypatch.setattr(router, "extract_requirements", lambda *a, **k: {"required_skills": []})
    monkeypatch.setattr(router, "_reqs_cache_put", lambda *a: None)
    fb = {}
    _r, _hit, method = router._get_requirements(
        AgentState(goal="g"), {"title": "t", "description": "d"}, 1, 1, fallback=fb)
    assert method == "llm" and fb == {}


@pytest.fixture
def decide_ctx(monkeypatch):
    import agent_loop
    import agent_store
    import llm
    from agent_goal import AgentGoal
    monkeypatch.setattr(agent_loop.run_lock, "check_owner", lambda rid: None)
    monkeypatch.setattr(agent_store, "is_cancel_requested", lambda rid: False)
    monkeypatch.setattr(agent_loop, "check_limits", lambda *a, **k: None)
    monkeypatch.setattr(agent_store, "get_action", lambda rid, i: None)
    monkeypatch.setattr(agent_loop, "qualified_count", lambda ctx: 0)
    monkeypatch.setattr(llm, "quota_blocked", lambda: 0)
    monkeypatch.setattr(llm, "create_step", lambda *a, **k: 11)
    monkeypatch.setattr(llm, "fail_step", lambda sid, e: None)
    monkeypatch.setattr(llm, "finish_step", lambda *a, **k: None)
    recorded = []
    monkeypatch.setattr(agent_store, "record_proposed_action",
                        lambda *a, **k: recorded.append(k.get("fallback_code")))
    goal = AgentGoal(target_role="AI Engineer", use_llm_controller=True)
    state = {"run_id": 7, "goal": goal.model_dump(), "iteration": 1, "controller_mode": "llm",
             "resume_id": 1}
    return agent_loop, state, {"configurable": {"generation": 3}}, recorded


def test_controller_budget_spent_records_its_code(monkeypatch, decide_ctx):
    agent_loop, state, config, recorded = decide_ctx
    monkeypatch.setattr(agent_loop.DurableBudget, "exhausted", lambda self: True)
    out = agent_loop.node_decide(state, config)
    assert recorded == ["llm_budget_spent"] and out["controller_fallback"] == "llm_budget_spent"
    # the next iteration (mode now 'rules') keeps the recorded reason
    state2 = {**state, "iteration": 2, "controller_mode": out["controller_mode"],
              "controller_fallback": out["controller_fallback"]}
    agent_loop.node_decide(state2, config)
    assert recorded == ["llm_budget_spent", "llm_budget_spent"]


def test_controller_cost_cap_records_its_code(monkeypatch, decide_ctx):
    import llm
    agent_loop, state, config, recorded = decide_ctx
    monkeypatch.setattr(agent_loop.DurableBudget, "exhausted", lambda self: False)

    def capped(*a, **k):
        raise llm.CostLimitReached("cost limit: would exceed")
    monkeypatch.setattr(agent_loop, "llm_decide", capped)
    agent_loop.node_decide(state, config)
    assert recorded == ["cost_limit_reached"]


def test_controller_invalid_output_is_not_an_allowed_reason(monkeypatch, decide_ctx):
    from agent_controller import ControllerOutputInvalid
    agent_loop, state, config, recorded = decide_ctx
    monkeypatch.setattr(agent_loop.DurableBudget, "exhausted", lambda self: False)

    def invalid(*a, **k):
        raise ControllerOutputInvalid("bad action")
    monkeypatch.setattr(agent_loop, "llm_decide", invalid)
    agent_loop.node_decide(state, config)
    assert recorded == ["controller_output_invalid"]
    assert "controller_output_invalid" not in _load().ALLOWED_FALLBACK_CODES


def test_llm_decided_action_records_no_fallback_code(monkeypatch, decide_ctx):
    from agent_controller import ControllerDecision
    agent_loop, state, config, recorded = decide_ctx
    monkeypatch.setattr(agent_loop.DurableBudget, "exhausted", lambda self: False)
    monkeypatch.setattr(agent_loop, "llm_decide", lambda *a, **k: ControllerDecision(
        action="search_jobs", arguments={"query": "AI Engineer", "provider": "remotive"},
        reason="search the role"))
    agent_loop.node_decide(state, config)
    assert recorded == [None]


# --- Postgres: the record is read from real rows -----------------------------

def _uid():
    from auth import create_user
    return create_user("smf_" + uuid.uuid4().hex[:10], "password-1234")


@pytest.mark.db
def test_collect_run_record_reads_fallback_reasons_and_provider_recovery():
    from database import get_connection
    uid = _uid()
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                    "VALUES ('success', 't', %s, 'agent') RETURNING id, execution_generation",
                    (uid,))
        rid, gen = cur.fetchone()
        for i, (by, code) in enumerate([("rules", "cost_limit_reached"), ("rules", None),
                                        ("llm", None)], start=1):
            cur.execute("INSERT INTO agent_actions (run_id, iteration, execution_generation, "
                        "action, arguments, reason, decided_by, status, fallback_code) "
                        "VALUES (%s, %s, %s, 'rank_jobs', '{}', 'r', %s, 'executed', %s)",
                        (rid, i, gen, by, code))
        ctxs = [({"job_id": 1, "requirements_method": "rule_based",
                  "requirements_fallback_code": "llm_budget_spent"}, None),
                ({"job_id": 2, "requirements_method": "rule_based"}, None),
                ({"job_id": 3, "requirements_method": "llm",
                  "judge_fallback_code": "llm_unavailable"}, "unavailable"),
                ({"job_id": 4, "requirements_method": "llm"}, "ran")]
        for ctx, judge in ctxs:
            cur.execute("INSERT INTO steps (run_id, step_name, step_order, status, "
                        "retrieved_context, judge_status) VALUES (%s, 'job', 0, 'success', %s, %s)",
                        (rid, json.dumps(ctx), judge))
        # provider: q1 failed then succeeded (recovered); q2 failed only (unrecovered)
        for q, status in [("q1", "failed"), ("q1", "succeeded"), ("q2", "failed")]:
            cur.execute("INSERT INTO external_search_attempts (run_id, iteration, "
                        "execution_generation, provider, query_norm, status, finished_at) "
                        "VALUES (%s, 1, %s, 'remotive', %s, %s, NOW())", (rid, gen, q, status))
        # one agent_searches row per generation: the retry is the next generation's
        for g, status in [(gen, "failed"), (gen + 1, "success")]:
            cur.execute("INSERT INTO agent_searches (run_id, provider, query_norm, iteration, "
                        "provider_status, execution_generation) "
                        "VALUES (%s, 'remotive', 'q1', 1, %s, %s)", (rid, status, g))
    mod = _load()
    rec = mod.collect_run_record(rid)
    # 2 rules actions + 2 rule-based requirements + 1 unavailable judge
    assert rec["rules_fallbacks"] == 5
    assert rec["rules_fallback_reasons"] == {"cost_limit_reached": 1, "unrecorded": 2,
                                             "llm_budget_spent": 1, "llm_unavailable": 1}
    assert rec["provider_errors_unrecovered"] == 1
    assert rec["provider_errors"] == 3
    failed = [w for ok, w in mod.check_run(rec, 0.05) if not ok]
    assert any("rules fallback" in w for w in failed)
    assert any("unrecovered provider error" in w for w in failed)


@pytest.mark.db
def test_fallback_code_only_on_a_rules_decision():
    import psycopg2
    from database import get_connection
    uid = _uid()
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                    "VALUES ('running', 't', %s, 'agent') RETURNING id, execution_generation",
                    (uid,))
        rid, gen = cur.fetchone()
    with pytest.raises(psycopg2.errors.CheckViolation):
        with get_connection() as c:
            c.cursor().execute(
                "INSERT INTO agent_actions (run_id, iteration, execution_generation, action, "
                "arguments, reason, decided_by, status, fallback_code) "
                "VALUES (%s, 1, %s, 'rank_jobs', '{}', 'r', 'llm', 'proposed', 'x')", (rid, gen))
