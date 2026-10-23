"""Backlog E: runs paused before migration 0016 resume without requirement
snapshots. Advice then RECONSTRUCTS the requirements (requirements-cache hit or the
rules extractor, never a new model call), persists them as a snapshot tagged
provenance "reconstructed", and marks the advice as based on reconstructed
requirements in the trace and the API. The visible failure remains only when
reconstruction is impossible (posting gone or its text erased). Synthetic text only.
"""
import uuid

import pytest

import agent_store
import agent_tools
import llm
import resume_advisor
import router
from agent_goal import AgentGoal

GOAL = AgentGoal(target_role="Data Engineer", use_llm_advice=False)
DESC = "Data Engineer. Python and SQL required. 3+ years of experience required."


def _state(step_id=41):
    return {"evaluated": {"1": {"job_id": 1, "status": "ok", "final_decision": "Apply",
                                "step_id": step_id}},
            "resume_text": "Data engineer. Python and SQL.",
            "parsed_resume": {"skills": ["python", "sql"]}}


@pytest.fixture
def stub_env(monkeypatch):
    seen = {"reqs": [], "recorded": [], "llm": 0}
    store_rows = {}
    monkeypatch.setattr(agent_store, "advised_job_ids", lambda rid: [])
    monkeypatch.setattr(agent_store, "persist_advice", lambda *a, **k: None)
    monkeypatch.setattr(llm, "create_step", lambda *a, **k: 9)
    monkeypatch.setattr(llm, "finish_step", lambda *a, **k: None)
    monkeypatch.setattr(llm, "fail_step", lambda *a, **k: None)
    monkeypatch.setattr(router, "_reqs_cache_get", lambda h: (None, None))

    def no_llm(*a, **k):
        seen["llm"] += 1
        raise AssertionError("reconstruction must never call the model")
    monkeypatch.setattr(router, "extract_requirements", no_llm, raising=False)
    import job_parser
    monkeypatch.setattr(job_parser, "extract_requirements", no_llm)

    def record(rid, gen, sid, jid, reqs, prov):
        seen["recorded"].append((rid, gen, sid, jid, prov))
        store_rows.setdefault(sid, (reqs, prov))
    monkeypatch.setattr(agent_store, "record_evaluation_requirements", record)
    monkeypatch.setattr(agent_store, "evaluation_requirements",
                        lambda rid, sid: store_rows.get(sid))

    def build(parsed, text, job, reqs, sc, **k):
        seen["reqs"].append(reqs)
        return [], "rules"
    monkeypatch.setattr(resume_advisor, "build_suggestions", build)
    return seen


def _postings(monkeypatch, desc):
    monkeypatch.setattr(agent_store, "load_postings",
                        lambda ids: {i: {"id": i, "title": "Data Engineer", "company": "Acme",
                                         "description": desc} for i in ids})


def test_missing_snapshot_is_reconstructed_by_rules(stub_env, monkeypatch):
    _postings(monkeypatch, DESC)
    state = _state()
    ctx = agent_tools.ToolContext(1, 3, GOAL, state, None, 4)
    obs, _sid, _p = agent_tools.execute(ctx, "generate_advice", {"job_ids": [1]})
    assert obs["advised"] == [1] and obs["failed_job_ids"] == []
    assert obs["requirements_reconstructed"] == [1]
    (rid, gen, sid, jid, prov), = stub_env["recorded"]
    assert (rid, gen, sid, jid) == (1, 3, 41, 1)          # fenced, bound to the eval step
    assert prov["provenance"] == "reconstructed" and prov["source"] == "rules"
    assert prov["extraction_method"] == "rule_based"
    reqs = stub_env["reqs"][0]
    assert "python" in reqs["required_skills"] and reqs["min_years_experience"] == 3.0
    assert stub_env["llm"] == 0


def test_cache_hit_is_used_for_reconstruction(stub_env, monkeypatch):
    _postings(monkeypatch, DESC)
    cached = {"required_skills": ["python", "sql", "airflow"], "required_any_of": [],
              "preferred_skills": [], "min_years_experience": 3, "responsibilities": []}
    monkeypatch.setattr(router, "_reqs_cache_get",
                        lambda h: (cached, {"extraction_method": "llm", "source_model": "m"}))
    ctx = agent_tools.ToolContext(1, 3, GOAL, _state(), None, 4)
    obs, _sid, _p = agent_tools.execute(ctx, "generate_advice", {"job_ids": [1]})
    assert obs["advised"] == [1]
    prov = stub_env["recorded"][0][4]
    assert prov["source"] == "cache" and prov["extraction_method"] == "llm"
    assert "airflow" in stub_env["reqs"][0]["required_skills"]


@pytest.mark.parametrize("desc", ["", "   "])
def test_erased_posting_text_still_fails_visibly(stub_env, monkeypatch, desc):
    _postings(monkeypatch, desc)
    state = _state()
    ctx = agent_tools.ToolContext(1, 3, GOAL, state, None, 4)
    obs, _sid, _p = agent_tools.execute(ctx, "generate_advice", {"job_ids": [1]})
    assert obs["failed_job_ids"] == [1] and obs["advised"] == []
    assert obs["generation"]["1"] == "requirements_snapshot_missing"
    assert stub_env["reqs"] == [] and stub_env["recorded"] == []
    assert state["output_failures"] == 1


def test_missing_posting_fails_visibly(stub_env, monkeypatch):
    monkeypatch.setattr(agent_store, "load_postings", lambda ids: {})
    ctx = agent_tools.ToolContext(1, 3, GOAL, _state(), None, 4)
    obs, _sid, _p = agent_tools.execute(ctx, "generate_advice", {"job_ids": [1]})
    assert obs["failed_job_ids"] == [1]
    assert obs["generation"]["1"] == "posting_missing"
    assert stub_env["recorded"] == []


def test_evaluation_without_step_id_fails_visibly(stub_env, monkeypatch):
    _postings(monkeypatch, DESC)
    ctx = agent_tools.ToolContext(1, 3, GOAL, _state(step_id=None), None, 4)
    obs, _sid, _p = agent_tools.execute(ctx, "generate_advice", {"job_ids": [1]})
    assert obs["failed_job_ids"] == [1]
    assert obs["generation"]["1"] == "requirements_snapshot_missing"


# ----------------------------------------------------------------- Postgres --

def _legacy_run(description):
    """A run as it looked when paused before 0016: a successful evaluation step for
    the job, but no evaluation_requirements row."""
    from auth import create_user
    from database import get_connection
    uid = create_user("ar_" + uuid.uuid4().hex[:10], "password-1234")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode, started_at) "
                    "VALUES ('running', 't', %s, 'agent', NOW()) RETURNING id", (uid,))
        rid = cur.fetchone()[0]
        cur.execute("INSERT INTO job_postings (title, company, description, source, external_id,"
                    " last_seen_at, created_at) VALUES ('Data Engineer', 'Acme', %s, 'adzuna', %s,"
                    " NOW(), NOW()) RETURNING id",
                    (description, "adzuna:ar-" + uuid.uuid4().hex[:10]))
        jid = cur.fetchone()[0]
        cur.execute("INSERT INTO steps (run_id, step_name, step_order, status, final_decision,"
                    " retrieved_context) VALUES (%s, 'Data Engineer', 0, 'success', 'Apply', %s)"
                    " RETURNING id", (rid, '{"job_id": %d}' % jid))
        sid = cur.fetchone()[0]
        cur.execute("INSERT INTO resumes (name, resume_text, created_at, user_id) "
                    "VALUES ('synthetic', 'Data engineer. Python and SQL.', NOW(), %s) "
                    "RETURNING id", (uid,))
        _RESUMES[rid] = cur.fetchone()[0]
    gen = agent_store.begin_execution(rid, new_attempt=True)
    return uid, rid, gen, jid, sid


_RESUMES = {}


def _advise(rid, gen, jid, sid):
    state = {"resume_id": _RESUMES[rid], "evaluated": {str(jid): {"job_id": jid, "status": "ok", "final_decision": "Apply",
                                      "step_id": sid, "title": "Data Engineer"}},
             "resume_text": "Data engineer. Python and SQL. 4 years of experience.",
             "parsed_resume": {"skills": ["python", "sql"]}}
    ctx = agent_tools.ToolContext(rid, gen, GOAL, state, None, 4)
    return agent_tools.execute(ctx, "generate_advice", {"job_ids": [jid]})[0]


@pytest.mark.db
def test_legacy_run_gets_reconstructed_advice(monkeypatch):
    import api
    monkeypatch.setattr(router, "_reqs_cache_get", lambda h: (None, None))
    uid, rid, gen, jid, sid = _legacy_run(DESC)
    assert agent_store.evaluation_requirements(rid, sid) is None
    obs = _advise(rid, gen, jid, sid)
    assert obs["advised"] == [jid] and obs["requirements_reconstructed"] == [jid]
    reqs, prov = agent_store.evaluation_requirements(rid, sid)
    assert prov["provenance"] == "reconstructed" and prov["source"] == "rules"
    assert "python" in reqs["required_skills"]
    assert jid in agent_store.advised_job_ids(rid)
    # Visible in the API next to the advice.
    agent_store.persist_rankings(rid, gen, [{"job_id": jid, "title": "Data Engineer",
                                             "company": "Acme", "score": 80,
                                             "final_decision": "Apply"}])
    row = api.get_run_rankings(rid, user={"id": uid})["rankings"][0]
    assert row["advice"] and row["advice_requirements"] == "reconstructed"
    # ... and in the action trace observation (ids only).
    assert api._redact_tree(obs)["requirements_reconstructed"] == [jid]


@pytest.mark.db
def test_regular_snapshot_is_labeled_evaluation_in_api():
    import api
    uid, rid, gen, jid, sid = _legacy_run(DESC)
    agent_store.record_evaluation_requirements(
        rid, gen, sid, jid, {"required_skills": ["python"], "required_any_of": [],
                             "preferred_skills": [], "min_years_experience": 0,
                             "responsibilities": []}, {"extraction_method": "llm"})
    obs = _advise(rid, gen, jid, sid)
    assert obs["advised"] == [jid] and "requirements_reconstructed" not in obs
    agent_store.persist_rankings(rid, gen, [{"job_id": jid, "title": "Data Engineer",
                                             "company": "Acme", "score": 80,
                                             "final_decision": "Apply"}])
    row = api.get_run_rankings(rid, user={"id": uid})["rankings"][0]
    assert row["advice_requirements"] == "evaluation"


@pytest.mark.db
def test_erased_posting_text_fails_visibly_in_db():
    uid, rid, gen, jid, sid = _legacy_run("")
    obs = _advise(rid, gen, jid, sid)
    assert obs["failed_job_ids"] == [jid] and obs["advised"] == []
    assert obs["generation"][str(jid)] == "requirements_snapshot_missing"
    assert agent_store.evaluation_requirements(rid, sid) is None
