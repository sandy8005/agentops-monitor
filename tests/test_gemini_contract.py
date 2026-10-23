"""
Backlog F (2): Gemini contract tests. No network and no real key: the SDK client
(llm.get_client) is replaced by a fake that returns RECORDED provider outcomes:
response objects shaped like google-genai's GenerateContentResponse, and the SDK's
own exception classes (google.genai.errors.ClientError / ServerError, httpx
timeouts). Everything above the client is the real code: llm.real_llm_once,
llm.logged_llm_call (retries, classification, ledger) and each prompt's caller.

For every prompt type (parse, requirements, judge, evaluator, advice, controller)
and every scenario (valid, malformed JSON, missing fields, safety-blocked, 429,
5xx, timeout) the test asserts:
  * the classification: a failure is a degraded-model error
    (llm.is_degraded_model_error) and its ErrorCode;
  * the caller's outcome: the model result, or its rules fallback;
  * (db) the cost-ledger rows: one per HTTP attempt with the right cost_status,
    every reservation settled, and cost_ledger_violations(run_id) empty.
Synthetic text only.
"""
import json
import uuid
from types import SimpleNamespace as NS

import httpx
import pytest
from google.genai import errors as genai_errors

import llm
from error_codes import ErrorCode, classify_exception

RESUME = ("Sam Synthetic. Python engineer with 5 years of experience. "
          "Built a payments API in Python and Django. Ran PostgreSQL on AWS.")
JOB = {"id": 7, "title": "Data Engineer", "company": "Acme",
       "description": "Python and SQL required. 3+ years of experience required."}


# ------------------------------------------------------------ recorded data --

def _usage(prompt=120, cand=40, thoughts=None):
    return NS(prompt_token_count=prompt, candidates_token_count=cand,
              thoughts_token_count=thoughts, tool_use_prompt_token_count=None)


def _response(text, usage=None, block=None, finish=None):
    """Shape of google.genai.types.GenerateContentResponse as real_llm_once reads it."""
    return NS(text=text, usage_metadata=usage if usage is not None else _usage(),
              prompt_feedback=NS(block_reason=block),
              candidates=[NS(finish_reason=finish)] if finish else [],
              response_id="recorded-" + uuid.uuid4().hex[:8])


def _err_429():
    return genai_errors.ClientError(429, {"error": {
        "code": 429, "message": "Too many requests, please slow down.",
        "status": "RESOURCE_EXHAUSTED"}})


def _err_500():
    return genai_errors.ServerError(500, {"error": {
        "code": 500, "message": "An internal error has occurred.", "status": "INTERNAL"}})


def _err_503():
    return genai_errors.ServerError(503, {"error": {
        "code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}})


def _timeout():
    return httpx.ReadTimeout("")          # what the SDK's httpx transport raises


VALID = {
    "parse": {"skills": ["Python", "Django", "PostgreSQL", "AWS"], "years_experience": 5,
              "education": [], "projects": [{"name": "payments API", "tech": ["Python", "Django"],
                                             "evidence": "Built a payments API in Python and Django"}],
              "experience": [{"title": "Python engineer", "company": "", "years": 5}],
              "skill_evidence": [{"skill": "Python", "evidence": "Python engineer"}]},
    "requirements": {"required_skills": ["Python", "SQL"], "required_any_of": [],
                     "preferred_skills": [], "min_years_experience": 3,
                     "responsibilities": ["build pipelines"]},
    "judge": {"decision": "Apply", "reason": "Python and SQL are covered."},
    "evaluator": {"relevance_score": 8, "faithfulness_score": 9, "completeness_score": 7,
                  "hallucination_detected": False, "hallucinated_claims": [],
                  "notes": "Grounded in the resume."},
    "advice": {"rewrites": [{"original_text": "Built a payments API in Python and Django",
                             "suggested_text": "Built a payments API in Python and Django, "
                                               "covering the posting's Python requirement",
                             "reason": "names the required skill"}]},
    "controller": {"action": "search_jobs",
                   "arguments": {"provider": "remotive", "query": "data engineer"},
                   "reason": "nothing has been searched yet"},
}
MISSING_FIELDS = {
    "parse": {},
    "requirements": {"responsibilities": ["x"], "min_years_experience": "many"},
    "judge": {"decision": "Apply"},
    "evaluator": {"relevance_score": 8},
    "advice": {"rewrites": [{"original_text": "Built a payments API in Python and Django"}]},
    "controller": {"arguments": {}},
}
PROMPT_TYPES = list(VALID)


def recorded(kind, scenario):
    """The provider outcome for (prompt type, scenario): a response or an exception."""
    if scenario == "valid":
        return _response(json.dumps(VALID[kind]))
    if scenario == "malformed_json":
        return _response('{"skills": ["Python", "decision": Apply,,}')
    if scenario == "missing_fields":
        return _response(json.dumps(MISSING_FIELDS[kind]))
    if scenario == "safety_blocked":
        return _response(None, usage=_usage(prompt=120, cand=0), block="SAFETY")
    return {"rate_limited_429": _err_429, "server_500": _err_500, "server_503": _err_503,
            "timeout": _timeout}[scenario]()


SCENARIOS = ["valid", "malformed_json", "missing_fields", "safety_blocked",
             "rate_limited_429", "server_500", "server_503", "timeout"]
PROVIDER_FAILURES = {"safety_blocked", "rate_limited_429", "server_500", "server_503", "timeout"}


class FakeModels:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def count_tokens(self, model, contents):           # never used (byte bound)
        raise AssertionError("no token-count request in contract tests")


@pytest.fixture
def fake_gemini(monkeypatch):
    """Install a fake client; records every ledger write instead of the DB."""
    rows = []
    state = {"models": None}

    def install(outcome, ledger=True):
        state["models"] = FakeModels(outcome)
        monkeypatch.setattr(llm, "get_client", lambda: NS(models=state["models"]))
        return state["models"]
    monkeypatch.setattr(llm, "_backoff_seconds", lambda attempt, exc=None: 0.0)
    monkeypatch.setattr(llm, "_log_llm_attempt",
                        lambda *a, **k: rows.append({"status": a[9],
                                                     "cost_status": k.get("cost_status")}))
    install.rows = rows
    return install


class _CountBudget:
    """Request-count budget without a reservation (the offline caller tests)."""
    def can_spend(self):
        return True

    def spend(self):
        pass


# ---------------------------------------------------------- classification --

EXPECTED_CODE = {"safety_blocked": ErrorCode.LLM_INVALID_RESPONSE,
                 "rate_limited_429": ErrorCode.LLM_RATE_LIMITED,
                 "server_500": ErrorCode.LLM_UNAVAILABLE,
                 "server_503": ErrorCode.LLM_UNAVAILABLE,
                 "timeout": ErrorCode.LLM_UNAVAILABLE}


@pytest.mark.parametrize("scenario", sorted(PROVIDER_FAILURES))
def test_provider_failures_are_degraded_and_classified(fake_gemini, scenario):
    models = fake_gemini(recorded("judge", scenario))
    with pytest.raises(Exception) as ei:
        llm.logged_llm_call("prompt", None, None, operation="job_judge", budget=_CountBudget())
    e = ei.value
    assert llm.is_degraded_model_error(e)
    assert classify_exception(e) == EXPECTED_CODE[scenario]
    # transient failures are retried up to the HTTP attempt cap; a safety block is not
    expected_attempts = 1 if scenario == "safety_blocked" else llm.LLM_HTTP_MAX_ATTEMPTS
    assert models.calls == expected_attempts
    assert len(fake_gemini.rows) == expected_attempts
    # a provider failure is never a programming error the policy would re-raise
    import router
    assert not router.must_propagate(e)


def test_safety_block_carries_reported_usage(fake_gemini):
    fake_gemini(recorded("judge", "safety_blocked"))
    with pytest.raises(llm.InvalidProviderResponse) as ei:
        llm.real_llm_once("prompt")
    assert (ei.value.prompt_tokens, ei.value.completion_tokens) == (120, 0)


# ------------------------------------------------------------ per prompt type --

class _ParseState:
    model_policy = "auto"
    completed_actions = []

    def __init__(self):
        self.resume_text = RESUME
        self.parsed_resume = None
        self.error = None

    def budget_exceeded(self):
        return False

    def can_spend(self):
        return True

    def spend(self):
        pass


def _run_parse(monkeypatch):
    import router
    cached = []
    monkeypatch.setattr(router, "create_step", lambda *a, **k: 1)
    monkeypatch.setattr(router, "finish_step", lambda *a, **k: None)
    monkeypatch.setattr(router, "fail_step", lambda *a, **k: None)
    monkeypatch.setattr(router, "_parse_cache_get", lambda h: None)
    monkeypatch.setattr(router, "_parse_cache_put", lambda h, p: cached.append(p))
    import prompt_safety
    monkeypatch.setattr(prompt_safety, "apply_injection_policy", lambda *a, **k: None)
    state = _ParseState()
    router.do_parse_resume(state, None)
    assert state.error is None and isinstance(state.parsed_resume, dict)
    return "model" if cached else "rules", state.parsed_resume


def _run_requirements(monkeypatch):
    import router
    monkeypatch.setattr(router, "_reqs_cache_get", lambda h: (None, None))
    monkeypatch.setattr(router, "_reqs_cache_put", lambda *a: None)
    reqs, _hit, method = router._get_requirements(_ParseState(), dict(JOB), None, None)
    assert isinstance(reqs, dict)
    return "model" if method == "llm" else "rules", reqs


def _run_judge(monkeypatch):
    import router
    raw = llm.logged_llm_call("judge prompt", None, None, operation="job_judge",
                              budget=_CountBudget())
    return "model", router._parse_decision(raw)


def _run_evaluator(monkeypatch):
    from evaluator import evaluate_decision
    return "model", evaluate_decision(RESUME, dict(JOB), "Apply: Python and SQL.", None, None,
                                      budget=_CountBudget())


def _run_advice(monkeypatch):
    import resume_advisor
    passage = {"source": "resume", "offset": 0, "verified": True,
               "text": "Built a payments API in Python and Django"}
    rule = {"kind": "emphasize", "original_text": None, "suggested_text": "Mention SQL work",
            "reason": "required skill", "evidence": [passage], "method": "rules",
            "status": "needs_confirmation", "validation_notes": None}
    monkeypatch.setattr(resume_advisor, "rule_suggestions", lambda *a, **k: [dict(rule)])
    suggestions, note = resume_advisor.build_suggestions(
        {"skills": ["python"]}, RESUME, dict(JOB), VALID["requirements"], {}, use_llm=True,
        budget=_CountBudget())
    assert suggestions[0]["method"] == "rules"           # rule suggestions are never lost
    return ("model" if note == "rules + gemini wording" else "rules"), note


def _run_controller(monkeypatch):
    from agent_controller import llm_decide
    from agent_goal import AgentGoal
    goal = AgentGoal(target_role="Data Engineer", providers=["remotive"])
    d = llm_decide(goal, {}, 0, ["search_jobs", "finish"], {"iterations": 5, "searches": 2},
                   None, None, _CountBudget())
    return "model", d


RUNNERS = {"parse": _run_parse, "requirements": _run_requirements, "judge": _run_judge,
           "evaluator": _run_evaluator, "advice": _run_advice, "controller": _run_controller}
# Callers with an in-place rules fallback return "rules"; the others raise, and
# their caller's handler (router judge/evaluator blocks, agent_loop decide node)
# takes the fallback on exactly the exceptions asserted here.
IN_PLACE_FALLBACK = {"parse", "requirements", "advice"}


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("kind", PROMPT_TYPES)
def test_prompt_contract(fake_gemini, monkeypatch, kind, scenario):
    fake_gemini(recorded(kind, scenario))
    if scenario == "valid":
        source, result = RUNNERS[kind](monkeypatch)
        assert source == "model", (kind, result)
        return
    if kind in IN_PLACE_FALLBACK:
        source, result = RUNNERS[kind](monkeypatch)
        assert source == "rules", (kind, scenario, result)
        if kind == "advice":
            assert result == ("rules (gemini output invalid)"
                              if scenario in ("malformed_json", "missing_fields")
                              else f"rules (gemini unavailable: {result.split(': ')[-1]}")
        return
    with pytest.raises(Exception) as ei:
        RUNNERS[kind](monkeypatch)
    e = ei.value
    import router
    assert not router.must_propagate(e), (kind, scenario, type(e).__name__)
    if kind == "controller":
        from agent_controller import ControllerOutputInvalid
        import agent_loop
        if scenario in ("malformed_json", "missing_fields"):
            assert isinstance(e, ControllerOutputInvalid)
        else:
            assert agent_loop.controller_model_degraded(e), (scenario, type(e).__name__)
    else:
        # judge / evaluator: unusable output and provider failures are both
        # degraded-model errors (judge: flagged judge_invalid_output / unavailable;
        # evaluator: flagged evaluation_failed) — never a crash of the run.
        assert llm.is_degraded_model_error(e), (kind, scenario, type(e).__name__)
        if scenario in ("malformed_json", "missing_fields"):
            assert isinstance(e, llm.ModelOutputInvalid)


def test_judge_list_output_is_invalid_not_a_programming_error(fake_gemini):
    import router
    fake_gemini(_response("[1, 2]"))
    raw = llm.logged_llm_call("p", None, None, operation="job_judge", budget=_CountBudget())
    with pytest.raises(llm.ModelOutputInvalid):
        router._parse_decision(raw)


# --------------------------------------------------------------- the ledger --

LEDGER = {   # scenario -> (rows, status, cost_status)
    "valid": (1, "success", "priced"),
    "malformed_json": (1, "success", "priced"),        # the model answered; caller rejects
    "missing_fields": (1, "success", "priced"),
    "safety_blocked": (1, "failed", "priced"),         # usage reported -> priced
    "rate_limited_429": (llm.LLM_HTTP_MAX_ATTEMPTS, "failed", "not_billed"),
    "server_500": (llm.LLM_HTTP_MAX_ATTEMPTS, "failed", "unknown"),
    "server_503": (llm.LLM_HTTP_MAX_ATTEMPTS, "failed", "unknown"),
    "timeout": (llm.LLM_HTTP_MAX_ATTEMPTS, "failed", "unknown"),
}


@pytest.mark.db
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_ledger_rows_per_scenario(monkeypatch, scenario):
    import agent_loop
    import agent_store
    from auth import create_user
    from database import get_connection
    monkeypatch.setattr(llm, "_backoff_seconds", lambda attempt, exc=None: 0.0)
    models = FakeModels(recorded("judge", scenario))
    monkeypatch.setattr(llm, "get_client", lambda: NS(models=models))
    uid = create_user("gc_" + uuid.uuid4().hex[:10], "password-1234")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode, max_cost_usd, "
                    "llm_call_budget) VALUES ('queued', 't', %s, 'agent', 0.5, 40) RETURNING id",
                    (uid,))
        rid = cur.fetchone()[0]
    gen = agent_store.begin_execution(rid, new_attempt=True)
    budget = agent_loop.DurableBudget(rid, 40, max_cost_usd=0.5, generation=gen)
    try:
        llm.logged_llm_call("judge prompt", rid, None, operation="job_judge", budget=budget)
        raised = None
    except Exception as e:      # the classification is asserted above; here: the ledger
        raised = e
    n, status, cost_status = LEDGER[scenario]
    assert (raised is None) == (scenario in ("valid", "malformed_json", "missing_fields"))
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("""SELECT status, cost_status, cost_usd, cost_upper_bound_usd, reservation_id,
                              execution_generation, operation_name
                       FROM llm_calls WHERE run_id = %s ORDER BY id""", (rid,))
        rows = cur.fetchall()
        assert len(rows) == n == models.calls
        for st, cs, cost, bound, res_id, g, op in rows:
            assert (st, cs, g, op) == (status, cost_status, gen, "job_judge")
            assert res_id is not None                  # every attempt was reserved first
            if cs == "priced":
                assert cost is not None and cost >= 0 and bound is None
            elif cs == "unknown":
                assert cost is None and bound is not None and bound > 0
            else:
                assert cost is None
        cur.execute("SELECT status, count(*) FROM llm_cost_reservations WHERE run_id = %s "
                    "GROUP BY status", (rid,))
        assert dict(cur.fetchall()) == {"settled": n}
        cur.execute("SELECT problem FROM cost_ledger_violations(%s)", (rid,))
        assert cur.fetchall() == []
    usage = agent_store.run_usage(rid)
    assert usage["open_reservations"] == 0 and usage["llm_calls_reserved"] == n
    assert usage["unknown_cost_calls"] == (n if cost_status == "unknown" else 0)
