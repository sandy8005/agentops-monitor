"""Backlog D: work-mode verification fails closed. A posting (or legacy state) that
reaches search, evaluation, ranking or goal counting WITHOUT a work_mode_verified
flag is computed from its stated work mode, never defaulted to verified. A
"remote" inferred from the location text counts only when the location says
remote explicitly. Synthetic postings only."""
import json
import uuid

import pytest

import agent_store
import agent_tools
import llm
import router
from agent_goal import AgentGoal
from agent_tools import EvaluateArgs, SearchArgs, ToolContext, tool_evaluate_jobs, tool_search_jobs
from job_source import work_mode_status, work_mode_verified

REMOTE = AgentGoal(target_role="Data Engineer", providers=["remotive"],
                   constraints={"work_mode": "remote"})


@pytest.mark.parametrize("location,status", [
    ("Remote", "verified"),
    ("Remote - Europe", "verified"),
    ("US (Remote)", "verified"),
    ("Remote, Germany", "verified"),
    ("Remote / USA", "verified"),
    ("Fully remote", "verified"),
    ("Not remote", "unknown"),
    ("Non-remote", "unknown"),
    ("No remote work", "unknown"),
    ("Remote-friendly", "unknown"),
    ("Remote possible", "unknown"),
    ("Partially remote", "unknown"),
    ("Hybrid (remote 2 days)", "unknown"),
    ("Remote or onsite", "unknown"),
    ("London office, remote optional", "unknown"),
    ("Remoteville, TX", "unknown"),
    ("Berlin", "unknown"),
    ("", "unknown"),
])
def test_location_inference_only_when_explicit(location, status):
    assert work_mode_status({"work_mode": None, "location": location}, "remote") == status


@pytest.mark.parametrize("job,constraint,expected", [
    ({"work_mode": None}, "remote", False),             # missing flag -> unverified
    ({"work_mode": "unknown"}, "remote", False),        # legacy placeholder value
    ({}, "remote", False),
    ({"work_mode": "remote"}, "remote", True),          # computed from the stated mode
    ({"work_mode": "hybrid"}, "remote", False),
    ({"work_mode": None}, None, False),                 # no constraint known: stated mode only
    ({"work_mode": "onsite"}, None, True),
    ({"work_mode": None, "work_mode_verified": True}, "remote", True),   # explicit flag wins
    ({"work_mode": "remote", "work_mode_verified": False}, "remote", False),
    ({"work_mode": "remote", "work_mode_verified": "yes"}, None, True),   # non-bool: recomputed
    ({"work_mode": None, "work_mode_verified": "true"}, "remote", False),
])
def test_missing_flag_is_computed_not_defaulted(job, constraint, expected):
    assert work_mode_verified(job, constraint) is expected


class _Budget:
    def check_owner(self):
        pass


def test_search_candidate_without_flag_is_unverified(monkeypatch):
    jobs = [{"id": 1, "title": "Data Engineer", "company": "A", "source": "remotive",
             "work_mode": "remote"},
            {"id": 2, "title": "Data Engineer", "company": "B", "source": "remotive",
             "work_mode": None, "location": "Remote-friendly"}]
    monkeypatch.setattr(agent_tools, "_collect_candidates", lambda ctx, p, q: jobs)
    monkeypatch.setattr(agent_tools, "_fetch_with_attempt", lambda *a: "success")
    monkeypatch.setattr(agent_store, "search_history", lambda *a: [])
    monkeypatch.setattr(agent_store, "record_search", lambda *a, **k: None)
    monkeypatch.setattr(llm, "create_step", lambda *a, **k: 1)
    monkeypatch.setattr(llm, "finish_step", lambda *a, **k: None)
    state = {"searches": [], "discovered": {}}
    obs, _, _ = tool_search_jobs(ToolContext(1, 1, REMOTE, state, _Budget(), 1),
                                 SearchArgs(provider="remotive", query="data engineer"))
    assert state["discovered"]["1"]["work_mode_verified"] is True
    assert state["discovered"]["2"]["work_mode"] == "unknown"
    assert state["discovered"]["2"]["work_mode_verified"] is False
    assert obs["unknown_work_mode"] == 1


@pytest.mark.parametrize("posting_mode,legacy_meta,expected_mode,expected_verified", [
    (None, {"work_mode": "unknown"}, "unknown", False),
    (None, {}, "unknown", False),                       # oldest checkpoints: no keys at all
    ("remote", {}, "remote", True),                     # computed from the stored posting
    (None, {"work_mode": "remote"}, "unknown", False),  # loose legacy label not trusted
])
def test_legacy_discovered_entry_without_flag(monkeypatch, posting_mode, legacy_meta,
                                              expected_mode, expected_verified):
    seen = []
    monkeypatch.setattr(agent_store, "load_postings",
                        lambda ids: {2: {"id": 2, "title": "Data Engineer", "company": "B",
                                         "description": "Python.", "work_mode": posting_mode,
                                         "location": "Berlin"}})
    monkeypatch.setattr(agent_tools, "_existing_evaluation", lambda rid, jid: None)
    monkeypatch.setattr(agent_tools, "qualified_count", lambda ctx: 0)

    def process(ts, run_id):
        seen.append(dict(ts.jobs[0]))
        ts.job_results.append({"step_id": 5, "job_id": 2, "title": "Data Engineer",
                               "company": "B", "score": 90.0, "decision": "Apply",
                               "final_decision": "Apply", "needs_review": False})
    monkeypatch.setattr(router, "do_process_job", process)
    meta = {"id": 2, "title": "Data Engineer", "company": "B", "eligible": True, **legacy_meta}
    state = {"discovered": {"2": meta}, "evaluated": {}}
    tool_evaluate_jobs(ToolContext(1, 1, REMOTE, state, _Budget(), 2), EvaluateArgs(job_ids=[2]))
    assert seen[0]["work_mode_verified"] is expected_verified
    assert state["evaluated"]["2"]["work_mode"] == expected_mode
    assert state["evaluated"]["2"]["work_mode_verified"] is expected_verified


# ----------------------------------------------------------------- Postgres --

def _db_run():
    from auth import create_user
    from database import get_connection
    uid = create_user("wmf_" + uuid.uuid4().hex[:10], "password-1234")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                    "VALUES ('running', 't', %s, 'agent') RETURNING id, execution_generation",
                    (uid,))
        rid, gen = cur.fetchone()
    return uid, rid, gen


def _posting(work_mode):
    from database import get_connection
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO job_postings (title, company, description, location, work_mode,"
                    " source, external_id) VALUES ('Data Engineer', 'Acme', 'Python.', 'Berlin',"
                    " %s, 'remotive', %s) RETURNING id", (work_mode, "wmf-" + uuid.uuid4().hex))
        return cur.fetchone()[0]


def _step(run_id, job_id, ctx_extra):
    from database import get_connection
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO steps (run_id, step_name, step_order, status, final_decision,"
                    " retrieved_context) VALUES (%s, 'job', 0, 'success', 'Apply', %s)",
                    (run_id, json.dumps({"job_id": job_id, **ctx_extra})))


@pytest.mark.db
def test_legacy_step_without_flag_counts_only_when_posting_states_remote():
    _uid, rid, _gen = _db_run()
    remote_id, unknown_id, flagged_id = _posting("remote"), _posting(None), _posting(None)
    _step(rid, remote_id, {})                               # legacy, posting says remote
    _step(rid, unknown_id, {})                              # legacy, posting states nothing
    _step(rid, flagged_id, {"work_mode_verified": True})    # explicit flag
    assert agent_store.qualified_job_ids(rid, ["Apply"]) == sorted([remote_id, unknown_id, flagged_id])
    assert agent_store.qualified_job_ids(rid, ["Apply"], require_work_mode_verified=True) == \
        sorted([remote_id, flagged_id])


@pytest.mark.db
def test_ranking_row_without_flag_is_stored_unverified():
    import api
    uid, rid, gen = _db_run()
    agent_store.persist_rankings(rid, gen, [
        {"job_id": 1, "title": "Data Engineer", "company": "A", "score": 90,
         "final_decision": "Apply", "work_mode": "remote"}])
    out = api.get_run_rankings(rid, user={"id": uid})["rankings"]
    assert out[0]["work_mode_verified"] is False
