"""Backlog F: cost is never "complete" while an LLM reservation is open.

$0.10 recorded + a $0.50 open reservation used to report total_cost = $0.10,
cost_complete = True. Open reservations are now reported separately and the
total is provisional until they settle.
"""
import uuid

import pytest

import api


def test_open_reservation_makes_cost_provisional():
    f = api._cost_fields(0.10, 0, 0.50, open_reservations=1, open_usd=0.50)
    assert f["cost_complete"] is False and f["total_cost"] is None
    assert f["cost_provisional"] is True
    assert f["open_reservations"] == 1 and f["provisional_usd"] == pytest.approx(0.50)
    assert f["known_cost_usd"] == pytest.approx(0.10)
    assert f["cost_upper_bound_usd"] == pytest.approx(0.60)


def test_settled_cost_is_complete():
    f = api._cost_fields(0.10, 0, 0.0)
    assert f["cost_complete"] is True and f["total_cost"] == pytest.approx(0.10)
    assert f["cost_provisional"] is False and f["open_reservations"] == 0


def _sql(q, args=()):
    from database import get_connection
    with get_connection() as c:
        cur = c.cursor()
        cur.execute(q, args)
        return cur.fetchone() if cur.description else None


@pytest.mark.db
def test_endpoints_report_open_reservation_as_provisional():
    import agent_store
    from auth import create_user
    uid = create_user("cp_" + uuid.uuid4().hex[:10], "password-1234")
    rid = _sql("INSERT INTO runs (status, input_summary, user_id, mode, started_at) "
               "VALUES ('running', 't', %s, 'agent', NOW()) RETURNING id", (uid,))[0]
    gen = agent_store.begin_execution(rid, new_attempt=True)
    _sql("INSERT INTO llm_calls (run_id, prompt_tokens, completion_tokens, cost_usd, status, "
         "cost_status) VALUES (%s, 10, 5, 0.10, 'success', 'priced')", (rid,))
    _sql("INSERT INTO llm_cost_reservations (run_id, execution_generation, operation, amount_usd)"
         " VALUES (%s, %s, 'job_judge', 0.50)", (rid, gen))

    usage = agent_store.run_usage(rid)
    assert usage["open_reservations"] == 1
    listed = [r for r in api.list_runs(limit=100, user={"id": uid}) if r["id"] == rid][0]
    assert listed["cost_complete"] is False and listed["total_cost"] is None
    assert listed["open_reservations"] == 1
    assert listed["provisional_usd"] == pytest.approx(0.50)
    timeline = api.get_agent_timeline(rid, user={"id": uid})["cost"]
    assert timeline["complete"] is False and timeline["open_reservations"] == 1
