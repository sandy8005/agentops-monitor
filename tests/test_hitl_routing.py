"""
Human-in-the-loop routing: a job pauses for review when it's flagged for ANY reason
(score disagreement, prompt injection, hallucination), decided from the AUTHORITATIVE
DB flag after every trigger — not just the score-vs-LLM disagreement that record_score
returns. The regression this guards: an injection-flagged job used to slip through when
the score and the judge happened to agree. Needs Postgres; the LLM / requirements /
evaluator are mocked so only the routing decision is under test.
"""
import uuid

import pytest

pytestmark = [pytest.mark.db]

from database import get_connection
from auth import create_user
from agent_state import AgentState
import router
from llm import flag_for_review


def _run(uid):
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (started_at,status,input_summary,user_id) "
                    "VALUES (NOW(),'running','t',%s) RETURNING id", (uid,))
        return cur.fetchone()[0]


def _state(evaluate=False):
    s = AgentState(goal="x", resume_id=1, target_role="engineer", evaluate=evaluate)
    s.resume_text = "Python SQL engineer resume"
    s.parsed_resume = {"skills": ["python"], "years_experience": 5,
                       "education": [], "projects": [], "experience": []}
    s.jobs = [{"id": None, "title": "Engineer", "company": "Acme",
               "description": "Build things with Python", "source": "seed",
               "apply_url": "https://apply.example/1"}]
    s.current_job_index = 0
    return s


def _wire(mp, *, score_decision, llm_decision, flag_injection=False, halluc=False):
    mp.setattr(router, "_reqs_cache_get", lambda dhash: (None, None))  # force extraction

    reqs = {"required_skills": ["python"], "required_any_of": [],
            "preferred_skills": [], "min_years_experience": 3}

    def fake_extract(job, run_id, step_id, budget=None):
        if flag_injection:  # simulate job_parser's prompt-injection detection
            flag_for_review(step_id, reason="possible_prompt_injection(job)")
        return reqs
    mp.setattr(router, "extract_requirements", fake_extract)

    mp.setattr(router, "calculate_match_score",
               lambda parsed, r, resume, job, ui: {
                   "score": 50, "decision": score_decision,   # mid-band -> judge runs
                   "matched_skills": ["python"], "missing_skills": [], "breakdown": {}})
    mp.setattr(router, "logged_llm_call", lambda *a, **k: "judge output")
    mp.setattr(router, "_parse_decision", lambda result: llm_decision)

    if halluc:
        import evaluator
        import llm as llmmod
        mp.setattr(evaluator, "evaluate_decision",
                   lambda resume, job, result, run_id, step_id, budget=None: {
                       "relevance_score": 5, "faithfulness_score": 5,
                       "completeness_score": 5, "hallucination_detected": True})
        mp.setattr(llmmod, "save_evaluation", lambda *a, **k: None)


def _run_job(mp, run_id, evaluate=False, **wire):
    _wire(mp, **wire)
    s = _state(evaluate=evaluate)
    router.do_process_job(s, run_id)
    assert len(s.job_results) == 1, "job did not complete the happy path (mock issue)"
    return s


def test_no_flags_continues(monkeypatch):
    uid = create_user("h1_" + uuid.uuid4().hex[:8], "password123")
    s = _run_job(monkeypatch, _run(uid), score_decision="Apply", llm_decision="Apply")
    assert s.last_job_needs_review is False
    assert s.job_results[-1]["needs_review"] is False


def test_score_disagreement_pauses(monkeypatch):
    uid = create_user("h2_" + uuid.uuid4().hex[:8], "password123")
    s = _run_job(monkeypatch, _run(uid), score_decision="Apply", llm_decision="Skip")
    assert s.last_job_needs_review is True


def test_prompt_injection_only_pauses(monkeypatch):
    # THE regression: injection flagged but score & judge AGREE — must still pause.
    uid = create_user("h3_" + uuid.uuid4().hex[:8], "password123")
    s = _run_job(monkeypatch, _run(uid), score_decision="Apply", llm_decision="Apply",
                 flag_injection=True)
    assert s.last_job_needs_review is True
    assert s.job_results[-1]["needs_review"] is True


def test_hallucination_pauses(monkeypatch):
    # A flagged (disagreeing) job runs the evaluator, which detects hallucination and
    # adds a flag; the final authoritative decision still pauses.
    uid = create_user("h4_" + uuid.uuid4().hex[:8], "password123")
    s = _run_job(monkeypatch, _run(uid), evaluate=True,
                 score_decision="Apply", llm_decision="Skip", halluc=True)
    assert s.last_job_needs_review is True


def test_multiple_reasons_pause(monkeypatch):
    uid = create_user("h5_" + uuid.uuid4().hex[:8], "password123")
    s = _run_job(monkeypatch, _run(uid), score_decision="Apply", llm_decision="Skip",
                 flag_injection=True)
    assert s.last_job_needs_review is True