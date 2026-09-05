"""
LangGraph orchestration of the autonomous agent — TypedDict state edition.

The graph state is now a FLAT, JSON-SERIALIZABLE TypedDict (not an AgentState
object), so it can be checkpointed reliably by a persistence backend. Each node
uses the ADAPTER pattern: hydrate an AgentState from the dict (from_dict), run the
existing, unchanged router tool (which mutates the object), then return the flat
dict (to_dict) for LangGraph to merge. router.py is UNTOUCHED — all its tested
logic (caching, judge signals, budget, cancellation checks) is reused as-is.

This is the foundation for LangGraph checkpointing + human-in-the-loop interrupts.
"""
import os
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

from agent_state import AgentState
from router import (
    load_resume, do_parse_resume, do_search_jobs, do_process_job,
    do_rank_jobs, do_generate_advice,
)
from llm import create_run, finish_run


def _db_uri():
    """
    libpq keyword/value connection string (NOT a URI). This avoids URI parsing
    entirely, so special characters in the password (@, #, :, /) are safe —
    a URI would mis-split on them. psycopg / PostgresSaver accept this format.
    """
    return (f"host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')} "
            f"dbname={os.getenv('DB_NAME')} user={os.getenv('DB_USER')} "
            f"password={os.getenv('DB_PASSWORD')}")


# --- Flat, serializable graph state. Every field is JSON-serializable so the
# checkpointer can persist it across a pause/resume. Mirrors AgentState's fields
# plus run_id. ---
class GraphState(TypedDict, total=False):
    run_id: int
    # identity / config
    goal: str
    resume_id: int
    target_role: Optional[str]
    location: Optional[str]
    work_mode: Optional[str]
    employment_type: Optional[str]
    evaluate: bool
    # accumulated results
    resume_text: Optional[str]
    parsed_resume: Optional[dict]
    jobs: Optional[list]
    current_job_index: int
    job_results: list
    ranked: Optional[list]
    # control flags
    ranking_done: bool
    advice_done: bool
    cancelled: bool
    done: bool
    error: Optional[str]
    # budget & failure accounting
    llm_calls_made: int
    max_llm_calls: int
    failed_jobs: int
    # misc
    requirements_cache: dict
    completed_actions: list


# --- Adapter helpers: dict <-> AgentState, so router.py stays unchanged. ---

def _hydrate(state: GraphState) -> AgentState:
    """Rebuild an AgentState from the flat graph-state dict."""
    return AgentState.from_dict(state)

def _dump(s: AgentState, run_id: int) -> dict:
    """Flatten an AgentState back to a serializable dict, carrying run_id."""
    d = s.to_dict()
    d["run_id"] = run_id
    return d


# --- Nodes: hydrate -> run existing tool -> return flat dict. ---

def node_load_resume(state: GraphState) -> dict:
    s = _hydrate(state)
    load_resume(s, state["run_id"])
    s.record_action("load_resume")
    return _dump(s, state["run_id"])

def node_parse_resume(state: GraphState) -> dict:
    s = _hydrate(state)
    do_parse_resume(s, state["run_id"])
    s.record_action("parse_resume")
    return _dump(s, state["run_id"])

def node_search_jobs(state: GraphState) -> dict:
    s = _hydrate(state)
    do_search_jobs(s, state["run_id"])
    s.record_action("search_jobs")
    return _dump(s, state["run_id"])

def node_process_job(state: GraphState) -> dict:
    s = _hydrate(state)
    do_process_job(s, state["run_id"])
    s.record_action("process_job")
    return _dump(s, state["run_id"])

def node_rank_jobs(state: GraphState) -> dict:
    s = _hydrate(state)
    do_rank_jobs(s, state["run_id"])
    s.record_action("rank_jobs")
    return _dump(s, state["run_id"])

def node_generate_advice(state: GraphState) -> dict:
    s = _hydrate(state)
    do_generate_advice(s, state["run_id"])
    s.record_action("generate_advice")
    return _dump(s, state["run_id"])


# --- Conditional routing: PURE PYTHON reading the flat dict. Zero LLM calls. ---

def route_after_start(state: GraphState) -> str:
    if state.get("error"):
        return "fail"
    return "parse_resume"

def route_after_parse(state: GraphState) -> str:
    if state.get("error"):
        return "fail"
    return "search_jobs"

def route_after_search(state: GraphState) -> str:
    if state.get("error"):
        return "fail"
    if not state.get("jobs"):            # honest empty — no jobs matched
        return "no_matches"
    return "process_job"

def route_after_process_job(state: GraphState) -> str:
    """Per-job LOOP: keep processing until all jobs done, then rank.
    NO budget short-circuit — over-budget jobs still get scored and skip the
    judge cleanly (matches the hand-rolled planner: budget gates Gemini, not work)."""
    if state.get("cancelled"):
        return "cancelled"
    if state.get("current_job_index", 0) < len(state.get("jobs") or []):
        return "process_job"
    return "rank_jobs"

def route_after_rank(state: GraphState) -> str:
    return "generate_advice"


def build_graph(checkpointer=None):
    g = StateGraph(GraphState)

    g.add_node("load_resume", node_load_resume)
    g.add_node("parse_resume", node_parse_resume)
    g.add_node("search_jobs", node_search_jobs)
    g.add_node("process_job", node_process_job)
    g.add_node("rank_jobs", node_rank_jobs)
    g.add_node("generate_advice", node_generate_advice)

    g.set_entry_point("load_resume")

    g.add_conditional_edges("load_resume", route_after_start,
                            {"parse_resume": "parse_resume", "fail": END})
    g.add_conditional_edges("parse_resume", route_after_parse,
                            {"search_jobs": "search_jobs", "fail": END})
    g.add_conditional_edges("search_jobs", route_after_search,
                            {"process_job": "process_job", "no_matches": END, "fail": END})
    g.add_conditional_edges("process_job", route_after_process_job,
                            {"process_job": "process_job", "rank_jobs": "rank_jobs", "cancelled": END})
    g.add_conditional_edges("rank_jobs", route_after_rank,
                            {"generate_advice": "generate_advice"})
    g.add_edge("generate_advice", END)

    # Compile WITH the checkpointer if provided — that's what enables state
    # persistence (and, later, interrupt/resume). Without it, a plain graph.
    return g.compile(checkpointer=checkpointer)


# NOTE: the graph is no longer compiled at import. The checkpointer holds a live
# DB connection scoped to a `with` block, so the graph is built per-run inside
# run_agent_graph. (A ConnectionPool could keep it warm later; simple first.)


def run_agent_graph(resume_id, target_role=None, location=None,
                    work_mode=None, employment_type=None, evaluate=False,
                    run_id=None, max_llm_calls=30):
    """
    LangGraph entry point — same signature as run_agent_autonomous.
    Builds the initial flat state, invokes the graph, derives the run status
    from the FINAL state dict, and finalizes the run.
    """
    if run_id is None:
        run_id = create_run("autonomous job search (langgraph)", resume_id=resume_id,
                            target_role=target_role, location=location,
                            work_mode=work_mode, employment_type=employment_type)

    # Build the initial flat state via AgentState (so defaults match exactly).
    seed = AgentState(
        goal="match resume to jobs", resume_id=resume_id,
        target_role=target_role, location=location,
        work_mode=work_mode, employment_type=employment_type, evaluate=evaluate,
    )
    seed.max_llm_calls = max_llm_calls
    initial = _dump(seed, run_id)

    final_status = "success"
    final_state = initial
    try:
        # Open the Postgres checkpointer for this run. setup() creates the
        # checkpoint tables on first use (idempotent). The graph is compiled
        # WITH the checkpointer so state is persisted per thread_id (= run_id).
        with PostgresSaver.from_conn_string(_db_uri()) as checkpointer:
            checkpointer.setup()
            graph = build_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": str(run_id)},
                      "recursion_limit": 100}
            final_state = graph.invoke(initial, config=config)
        if final_state.get("cancelled"):
            final_status = "cancelled"
        elif final_state.get("error"):
            final_status = "failed"
        elif final_state.get("jobs") is not None and len(final_state.get("jobs")) == 0:
            final_status = "no_matches"
        elif final_state.get("failed_jobs", 0) > 0:
            final_status = "completed_with_errors"
    except Exception as e:
        final_status = "failed"
        print(f"graph run failed: {e}")
    finally:
        finish_run(run_id, final_status)

    print(f"Run {run_id} (langgraph) finished: {final_status} "
          f"({final_state.get('llm_calls_made', 0)} LLM calls)")
    ranked = final_state.get("ranked")
    if ranked:
        print("\nRANKED JOBS:")
        for i, r in enumerate(ranked, 1):
            print(f"{i}. {r['title']} ({r['company']}) — "
                  f"score {r['score']} ({r['decision']}), judge: {r['llm_decision']}")
    return final_state


if __name__ == "__main__":
    import sys
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_agent_graph(resume_id=rid, target_role="engineer")