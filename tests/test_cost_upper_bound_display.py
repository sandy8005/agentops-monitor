"""Backlog C: whenever a run's cost is not complete, the API and the dashboard show
the known amount AND the full upper bound (known + unknown-call bounds + open
reservations), listing each source. A provisional run that also has unknown-cost
calls must not drop the bound or the unknown-call count.

Node is not available in CI, so formatCost/renderAgent are guarded by a static
read of static/app.js; the numbers are asserted on the API fields."""
import os
import re
import uuid

import pytest

import api

APP_JS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "app.js")


def test_fields_with_provisional_and_unknown_together():
    # known 0.10; 2 unknown calls bounded at 0.30; one open reservation of 0.50
    # (the SQL's unknown_bound includes the open reservation).
    f = api._cost_fields(0.10, 2, 0.30 + 0.50, open_reservations=1, open_usd=0.50)
    assert f["cost_complete"] is False and f["total_cost"] is None
    assert f["cost_provisional"] is True
    assert f["unknown_cost_calls"] == 2 and f["open_reservations"] == 1
    assert f["provisional_usd"] == pytest.approx(0.50)
    assert f["cost_upper_bound_usd"] == pytest.approx(0.90)


def _function_body(src, name):
    start = src.index("function " + name + "(")
    depth, i = 0, src.index("{", start)
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return src[i:j + 1]
    raise AssertionError("unterminated function " + name)


def test_format_cost_always_shows_upper_bound_and_sources():
    with open(APP_JS, encoding="utf-8") as fh:
        body = _function_body(fh.read(), "formatCost")
    # exactly one incomplete-cost rendering and the complete one: the provisional
    # case must not return early with only known + provisional
    assert body.count("return") == 2
    for field in ("known_cost_usd", "cost_upper_bound_usd", "unknown_cost_calls",
                  "open_reservations", "provisional_usd"):
        assert field in body, field
    assert "escapeHtml" in body


def test_render_agent_cost_line_shows_full_upper_bound():
    with open(APP_JS, encoding="utf-8") as fh:
        src = fh.read()
    line = src[src.index("LLM calls reserved"):src.index("Active runtime")]
    for field in ("known_estimated_usd", "upper_bound_usd", "calls_with_unknown_cost",
                  "open_reservations", "in_flight_reserved_usd"):
        assert re.search(r"\b" + field + r"\b", line), field


def _sql(q, args=()):
    from database import get_connection
    with get_connection() as c:
        cur = c.cursor()
        cur.execute(q, args)
        return cur.fetchone() if cur.description else None


@pytest.mark.db
def test_endpoints_upper_bound_includes_unknown_calls_and_open_reservations():
    import agent_store
    from auth import create_user
    uid = create_user("cub_" + uuid.uuid4().hex[:10], "password-1234")
    rid = _sql("INSERT INTO runs (status, input_summary, user_id, mode, started_at) "
               "VALUES ('running', 't', %s, 'agent', NOW()) RETURNING id", (uid,))[0]
    gen = agent_store.begin_execution(rid, new_attempt=True)
    _sql("INSERT INTO llm_calls (run_id, prompt_tokens, completion_tokens, cost_usd, status, "
         "cost_status) VALUES (%s, 10, 5, 0.10, 'success', 'priced')", (rid,))
    _sql("INSERT INTO llm_calls (run_id, prompt_tokens, completion_tokens, cost_usd, status, "
         "cost_status, cost_upper_bound_usd) VALUES (%s, 0, 0, NULL, 'failed', 'unknown', 0.30)",
         (rid,))
    _sql("INSERT INTO llm_cost_reservations (run_id, execution_generation, operation, amount_usd)"
         " VALUES (%s, %s, 'job_judge', 0.50)", (rid, gen))

    listed = [r for r in api.list_runs(limit=100, user={"id": uid}) if r["id"] == rid][0]
    assert listed["cost_complete"] is False and listed["cost_provisional"] is True
    assert listed["unknown_cost_calls"] == 1 and listed["open_reservations"] == 1
    assert listed["cost_upper_bound_usd"] == pytest.approx(0.90)
    detail = api.get_run(rid, user={"id": uid})
    assert detail["cost_upper_bound_usd"] == pytest.approx(0.90)
    timeline = api.get_agent_timeline(rid, user={"id": uid})["cost"]
    assert timeline["complete"] is False
    assert timeline["upper_bound_usd"] == pytest.approx(0.90)
    assert timeline["calls_with_unknown_cost"] == 1 and timeline["open_reservations"] == 1
