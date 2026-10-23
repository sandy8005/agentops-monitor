"""Backlog B: a search is credited only with the postings IT returned.

Query 1 returns a posting; query 2 (which the same posting also matches by role)
returns nothing. Query 2's observation must report 0 new jobs, not the posting
query 1 found. Run-wide deduplication stays a separate step. Needs Postgres;
synthetic postings only.
"""
import uuid

import pytest

pytestmark = [pytest.mark.db]

import agent_store
import agent_tools
import llm
from agent_goal import AgentGoal
from agent_tools import SearchArgs, ToolContext, _collect_candidates, tool_search_jobs
from auth import create_user
from database import get_connection
from job_search import associate_jobs, create_search


def _run():
    uid = create_user("si_" + uuid.uuid4().hex[:10], "password-1234")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                    "VALUES ('running', 't', %s, 'agent') RETURNING id", (uid,))
        return cur.fetchone()[0]


def _posting(source="adzuna", title="Python Engineer", company="Acme", url=None):
    eid = f"{source}:si-" + uuid.uuid4().hex[:10]
    with get_connection() as c:
        cur = c.cursor()
        cur.execute(
            """INSERT INTO job_postings
               (title, company, description, location, work_mode, employment_type,
                source, external_id, apply_url, last_seen_at, created_at)
               VALUES (%s, %s, 'Python role', 'USA', 'onsite', 'full-time',
                       %s, %s, %s, NOW(), NOW()) RETURNING id""",
            (title, company, source, eid, url or f"https://apply.example/{uuid.uuid4().hex}"))
        return cur.fetchone()[0]


def _searched(run_id, role, job_ids, provider="adzuna"):
    conn = get_connection()
    try:
        sid = create_search(run_id, role, "USA", provider, conn=conn)
        associate_jobs(sid, job_ids, conn=conn)
        conn.commit()
    finally:
        conn.close()


class _Budget:
    def check_owner(self):
        pass


def _ctx(run_id, state, iteration):
    goal = AgentGoal(target_role="Python Engineer", providers=["adzuna"],
                     constraints={"location": "USA"})
    return ToolContext(run_id, 1, goal, state, _Budget(), iteration)


@pytest.fixture
def no_side_effects(monkeypatch):
    recorded = []
    monkeypatch.setattr(agent_store, "search_history", lambda *a: [])
    monkeypatch.setattr(agent_store, "record_search",
                        lambda *a, **k: recorded.append(a[6]))
    monkeypatch.setattr(llm, "create_step", lambda *a, **k: 1)
    monkeypatch.setattr(llm, "finish_step", lambda *a, **k: None)
    monkeypatch.setattr(llm, "fail_step", lambda *a, **k: None)
    return recorded


def test_candidates_are_scoped_to_the_search_that_returned_them():
    run_id = _run()
    jid = _posting()
    _searched(run_id, "python engineer", [jid])
    ctx = _ctx(run_id, {}, 1)
    assert [j["id"] for j in _collect_candidates(ctx, "adzuna", "Python  Engineer")] == [jid]
    # Same posting matches the role "python developer", but that search returned nothing.
    assert _collect_candidates(ctx, "adzuna", "python developer") == []


def test_empty_second_query_reports_zero(monkeypatch, no_side_effects):
    run_id = _run()
    jid = _posting()
    state = {"searches": [], "discovered": {}}
    returns = {"python engineer": [jid], "python developer": []}

    def fetch(ctx, args, qnorm, step_id, plan):
        if returns[qnorm]:
            _searched(run_id, args.query, returns[qnorm])
            return "success"
        return "empty"
    monkeypatch.setattr(agent_tools, "_fetch_with_attempt", fetch)
    obs1, _, _ = tool_search_jobs(_ctx(run_id, state, 1),
                                  SearchArgs(provider="adzuna", query="python engineer"))
    assert obs1["new_jobs"] == 1
    obs2, _, _ = tool_search_jobs(_ctx(run_id, state, 2),
                                  SearchArgs(provider="adzuna", query="python developer"))
    assert obs2["new_jobs"] == 0 and obs2["duplicates"] == 0 and obs2["eligible_jobs"] == 0
    assert no_side_effects[-1]["new_jobs"] == 0


def test_run_wide_dedup_is_kept(monkeypatch, no_side_effects):
    run_id = _run()
    a = _posting(company="Globex", url="https://apply.example/same-req")
    # The same requisition reposted under a different posting id.
    b = _posting(company="Globex", url="https://apply.example/same-req")
    state = {"searches": [], "discovered": {}}
    returns = {"python engineer": [a], "python developer": [a, b]}

    def fetch(ctx, args, qnorm, step_id, plan):
        _searched(run_id, args.query, returns[qnorm])
        return "success"
    monkeypatch.setattr(agent_tools, "_fetch_with_attempt", fetch)
    tool_search_jobs(_ctx(run_id, state, 1), SearchArgs(provider="adzuna", query="python engineer"))
    obs2, _, _ = tool_search_jobs(_ctx(run_id, state, 2),
                                  SearchArgs(provider="adzuna", query="python developer"))
    assert obs2["new_jobs"] == 0 and obs2["duplicates"] >= 1
    assert set(state["discovered"]) == {str(a)}
