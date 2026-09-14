"""
End-to-end LangGraph integration test: a real run that PAUSES TWICE for human
review and is RESUMED through both interrupts to completion.

What is REAL here (the machinery under test):
  - the actual graph from autonomous_graph.build_graph() — real nodes, real
    conditional routing, the real human_review interrupt node, the real per-job
    review loop;
  - the real Postgres checkpointer (PostgresSaver), so state genuinely persists
    across each pause and is reloaded on resume;
  - the real resume path via Command(resume=...) driving the graph forward.

What is STUBBED (leaf tools only, to keep the test deterministic and offline):
  - load_resume / do_parse_resume / do_search_jobs seed a fixed 3-job state with
    no DB reads, no LLM, no Adzuna;
  - do_process_job flags the FIRST TWO jobs for review (so the graph pauses
    twice) and passes the third, advancing current_job_index each time — mirroring
    the real node's contract (sets last_job_needs_review, advances the index);
  - apply_human_decision / persistence become no-ops so no run row is required.

The graph, the interrupt/checkpoint/resume cycle, and the two-pause loop are NOT
mocked — that is the whole point of the test.

Skips cleanly when langgraph or the DB env is unavailable.
Run:  pytest test_langgraph_e2e_two_interrupts.py -v
"""
import os
import importlib
import pytest

# --- skip if langgraph isn't installed -------------------------------------
langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")
pytest.importorskip("langgraph.checkpoint.postgres",
                    reason="langgraph postgres checkpointer not installed")

from langgraph.types import Command
from langgraph.checkpoint.postgres import PostgresSaver

# --- skip if the DB isn't configured ---------------------------------------
_DB_VARS = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")


def _db_configured():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return all(os.getenv(v) for v in _DB_VARS)


pytestmark = pytest.mark.skipif(
    not _db_configured(),
    reason="Postgres env (DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD) not set",
)


def _db_uri():
    return (f"host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')} "
            f"dbname={os.getenv('DB_NAME')} user={os.getenv('DB_USER')} "
            f"password={os.getenv('DB_PASSWORD')}")


# ---------------------------------------------------------------------------
# Deterministic leaf-tool stubs. These patch the functions that autonomous_graph
# imported into its module namespace, so the REAL nodes call these instead of the
# LLM/DB/Adzuna versions. Graph structure, routing, interrupt, checkpointer and
# resume are all left untouched.
# ---------------------------------------------------------------------------

# Three jobs; the first two will be flagged, so the graph pauses twice.
_JOBS = [
    {"id": 1, "title": "AI Engineer", "company": "Acme", "description": "d1"},
    {"id": 2, "title": "ML Engineer", "company": "Beta", "description": "d2"},
    {"id": 3, "title": "Data Engineer", "company": "Gamma", "description": "d3"},
]


def _stub_load_resume(state, run_id):
    state.resume_text = "stub resume text"


def _stub_parse_resume(state, run_id):
    state.parsed_resume = {"projects": [], "years_experience": 3}


def _stub_search_jobs(state, run_id):
    state.jobs = [dict(j) for j in _JOBS]


def _stub_process_job(state, run_id):
    """
    Deterministic stand-in for the real do_process_job: flag the first TWO jobs
    for human review, pass the third, and ALWAYS advance the index (the real
    node's finally-clause contract). Records a result row like the real one.
    """
    idx = state.current_job_index
    job = state.jobs[idx]
    needs_review = idx < 2   # jobs 0 and 1 -> two interrupts; job 2 -> no pause

    state.job_results.append({
        "step_id": 1000 + idx, "job_id": job["id"],
        "title": job["title"], "company": job["company"],
        "score": 55.0, "decision": "Maybe", "llm_decision": "Maybe",
        "final_decision": "Maybe", "needs_review": needs_review,
        "apply_url": None,
    })
    state.last_job_needs_review = needs_review
    if needs_review:
        state.last_review_step_id = 1000 + idx
        state.last_review_info = {
            "step_id": 1000 + idx, "job_title": job["title"],
            "company": job["company"], "score": 55.0,
            "score_decision": "Maybe", "llm_decision": "Maybe",
        }
    state.current_job_index += 1


def _stub_apply_human_decision(state, run_id, step_id, decision, comment=""):
    # No DB write; just record the human's call in-memory like the real one does.
    state.human_decisions[str(step_id)] = {"decision": decision, "comment": comment}
    for r in state.job_results:
        if r.get("step_id") == step_id:
            r["final_decision"] = decision
            break


def _stub_rank_jobs(state, run_id):
    state.ranked = list(state.job_results)
    state.ranking_done = True


def _stub_generate_advice(state, run_id):
    state.advice_done = True


@pytest.fixture
def graph_module(monkeypatch):
    """Import autonomous_graph and patch its leaf tools in-place."""
    ag = importlib.import_module("autonomous_graph")
    monkeypatch.setattr(ag, "load_resume", _stub_load_resume)
    monkeypatch.setattr(ag, "do_parse_resume", _stub_parse_resume)
    monkeypatch.setattr(ag, "do_search_jobs", _stub_search_jobs)
    monkeypatch.setattr(ag, "do_process_job", _stub_process_job)
    monkeypatch.setattr(ag, "apply_human_decision", _stub_apply_human_decision)
    monkeypatch.setattr(ag, "do_rank_jobs", _stub_rank_jobs)
    monkeypatch.setattr(ag, "do_generate_advice", _stub_generate_advice)
    return ag


def test_two_interrupts_end_to_end(graph_module):
    """
    Drive the REAL graph: it must pause at the first flagged job, resume, pause
    again at the second, resume, then run to completion — with real checkpointing
    in between each pause.
    """
    ag = graph_module

    # Seed the initial flat state via the real AgentState so defaults match.
    from agent_state import AgentState
    seed = AgentState(goal="e2e", resume_id=1, target_role="engineer")
    initial = ag._dump(seed, run_id=987654)   # a run_id that need not exist (stubs skip DB)

    thread = "e2e-two-interrupts-test"
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 100}

    with PostgresSaver.from_conn_string(_db_uri()) as cp:
        cp.setup()
        graph = ag.build_graph(checkpointer=cp)

        # --- FIRST invoke: should PAUSE at interrupt #1 (job 0 flagged) ---
        r1 = graph.invoke(initial, config=config)
        assert isinstance(r1, dict) and r1.get("__interrupt__"), \
            f"expected first pause, got keys {list(r1.keys())}"
        payload1 = r1["__interrupt__"][0].value
        assert payload1.get("type") == "review_request"
        assert payload1.get("job_title") == "AI Engineer", payload1

        # State was checkpointed at the pause: exactly one job processed so far.
        snap1 = graph.get_state(config).values
        assert snap1["current_job_index"] == 1, snap1["current_job_index"]

        # --- RESUME #1: should CONTINUE, process job 1, and PAUSE at interrupt #2 ---
        r2 = graph.invoke(Command(resume={"decision": "Apply", "comment": "one"}),
                          config=config)
        assert isinstance(r2, dict) and r2.get("__interrupt__"), \
            f"expected SECOND pause, got keys {list(r2.keys())}"
        payload2 = r2["__interrupt__"][0].value
        assert payload2.get("type") == "review_request"
        assert payload2.get("job_title") == "ML Engineer", payload2

        # Checkpoint advanced: two jobs processed, first human decision recorded.
        snap2 = graph.get_state(config).values
        assert snap2["current_job_index"] == 2, snap2["current_job_index"]
        assert snap2["human_decisions"]["1000"]["decision"] == "Apply"

        # --- RESUME #2: should process job 2 (no pause) and COMPLETE ---
        r3 = graph.invoke(Command(resume={"decision": "Skip", "comment": "two"}),
                          config=config)
        assert not (isinstance(r3, dict) and r3.get("__interrupt__")), \
            f"expected completion, but paused again: {r3.get('__interrupt__')}"

        # Final state: all three jobs processed, both human decisions applied,
        # ranking + advice ran.
        assert r3["current_job_index"] == 3, r3["current_job_index"]
        assert r3["ranking_done"] is True
        assert r3["advice_done"] is True
        assert r3["human_decisions"]["1000"]["decision"] == "Apply"
        assert r3["human_decisions"]["1001"]["decision"] == "Skip"
        # The two flagged results carry their human decisions authoritatively.
        by_step = {r["step_id"]: r for r in r3["job_results"]}
        assert by_step[1000]["final_decision"] == "Apply"
        assert by_step[1001]["final_decision"] == "Skip"


def test_state_persists_across_the_two_pauses(graph_module):
    """
    Tighter assertion on the checkpoint contract: a fresh graph object built
    against the SAME thread_id sees the persisted state at each pause — proving
    the pauses go through the real checkpointer, not in-process memory.
    """
    ag = graph_module
    from agent_state import AgentState
    seed = AgentState(goal="e2e2", resume_id=1, target_role="engineer")
    initial = ag._dump(seed, run_id=987655)

    thread = "e2e-two-interrupts-persist"
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 100}

    with PostgresSaver.from_conn_string(_db_uri()) as cp:
        cp.setup()
        ag.build_graph(checkpointer=cp).invoke(initial, config=config)   # pause #1

        # A DIFFERENT graph instance, same checkpointer + thread, reads the pause.
        reader = ag.build_graph(checkpointer=cp)
        snap = reader.get_state(config)
        assert snap.values["current_job_index"] == 1
        assert snap.next, "expected the graph to be paused (has a next node)"

        # Resume to the second pause and re-read with yet another instance.
        reader.invoke(Command(resume={"decision": "Maybe"}), config=config)
        snap2 = ag.build_graph(checkpointer=cp).get_state(config)
        assert snap2.values["current_job_index"] == 2
        assert snap2.next, "expected the graph to be paused at the second review"