"""
LangGraph orchestration of the autonomous agent.

Design: LangGraph replaces the hand-rolled while-loop in autonomous_agent.py.
It ORCHESTRATES only — every routing decision is a pure-Python rule (your
planner logic), so LangGraph adds ZERO Gemini calls. All the real work stays
in the existing tool functions in router.py; nodes are thin wrappers that call
them. State is your existing AgentState, carried inside the graph state dict
and mutated in place (Option B — keeps all tool functions unchanged).
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END

from agent_state import AgentState
from router import (
    load_resume, do_parse_resume, do_search_jobs, do_process_job,
    do_rank_jobs, do_generate_advice,
)
from llm import create_run, finish_run


# --- Graph state: a single-key dict carrying your AgentState object. ---
class GraphState(TypedDict):
    state: AgentState
    run_id: int


# --- Nodes: thin wrappers around existing tools. Each mutates AgentState in
# place and returns the (same) state dict so LangGraph carries it forward. ---

def node_load_resume(gs: GraphState) -> GraphState:
    load_resume(gs["state"], gs["run_id"])
    gs["state"].record_action("load_resume")
    return gs

def node_parse_resume(gs: GraphState) -> GraphState:
    do_parse_resume(gs["state"], gs["run_id"])
    gs["state"].record_action("parse_resume")
    return gs

def node_search_jobs(gs: GraphState) -> GraphState:
    do_search_jobs(gs["state"], gs["run_id"])
    gs["state"].record_action("search_jobs")
    return gs

def node_process_job(gs: GraphState) -> GraphState:
    do_process_job(gs["state"], gs["run_id"])
    gs["state"].record_action("process_job")
    return gs

def node_rank_jobs(gs: GraphState) -> GraphState:
    do_rank_jobs(gs["state"], gs["run_id"])
    gs["state"].record_action("rank_jobs")
    return gs

def node_generate_advice(gs: GraphState) -> GraphState:
    do_generate_advice(gs["state"], gs["run_id"])
    gs["state"].record_action("generate_advice")
    return gs


# --- Conditional routing: PURE PYTHON (your planner rules). Zero LLM calls. ---

def route_after_start(gs: GraphState) -> str:
    """After load: error?  else parse."""
    s = gs["state"]
    if s.error:
        return "fail"
    return "parse_resume"

def route_after_parse(gs: GraphState) -> str:
    s = gs["state"]
    if s.error:
        return "fail"
    return "search_jobs"

def route_after_search(gs: GraphState) -> str:
    s = gs["state"]
    if s.error:
        return "fail"
    if not s.jobs:                       # honest empty — no jobs matched
        return "no_matches"
    return "process_job"

def route_after_process_job(gs: GraphState) -> str:
    """The per-job LOOP: keep processing until all jobs done, then rank.
    NO budget short-circuit — over-budget jobs still get scored and skip the
    judge cleanly (matches the hand-rolled planner: budget gates Gemini, not work)."""
    s = gs["state"]
    if s.cancelled:
        return "cancelled"
    if s.current_job_index < len(s.jobs):
        return "process_job"             # loop back to next job
    return "rank_jobs"

def route_after_rank(gs: GraphState) -> str:
    return "generate_advice"


def build_graph():
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
                            {"process_job": "process_job", "rank_jobs": "rank_jobs","cancelled": END})
    g.add_conditional_edges("rank_jobs", route_after_rank,
                            {"generate_advice": "generate_advice"})
    g.add_edge("generate_advice", END)

    return g.compile()


# Build once at import (compiling is cheap; reuse across runs).
GRAPH = build_graph()


def run_agent_graph(resume_id, target_role=None, location=None,
                    work_mode=None, employment_type=None, evaluate=False,
                    run_id=None, max_llm_calls=30):
    """
    LangGraph entry point — same signature as run_agent_autonomous, so the
    dashboard can call this instead. Returns the final AgentState.
    """
    state = AgentState(
        goal="match resume to jobs", resume_id=resume_id,
        target_role=target_role, location=location,
        work_mode=work_mode, employment_type=employment_type, evaluate=evaluate,
    )
    state.max_llm_calls = max_llm_calls

    if run_id is None:
        run_id = create_run("autonomous job search (langgraph)", resume_id=resume_id,
                            target_role=target_role, location=location,
                            work_mode=work_mode, employment_type=employment_type)

    final_status = "success"
    try:
        # recursion_limit guards against a routing bug looping forever (your #20
        # spirit, at the graph level). Generous but bounded.
        GRAPH.invoke({"state": state, "run_id": run_id},
                     config={"recursion_limit": 100})
        if state.cancelled:
            final_status = "cancelled"
        elif state.error:
            final_status = "failed"
        elif state.jobs is not None and len(state.jobs) == 0:
            final_status = "no_matches"
        elif state.failed_jobs > 0:
            final_status = "completed_with_errors"
    except Exception as e:
        final_status = "failed"
        print(f"graph run failed: {e}")
    finally:
        finish_run(run_id, final_status)

    print(f"Run {run_id} (langgraph) finished: {final_status} "
          f"({state.llm_calls_made} LLM calls)")
    if state.ranked:
        print("\nRANKED JOBS:")
        for i, r in enumerate(state.ranked, 1):
            print(f"{i}. {r['title']} ({r['company']}) — "
                  f"score {r['score']} ({r['decision']}), judge: {r['llm_decision']}")
    return state


if __name__ == "__main__":
    import sys
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_agent_graph(resume_id=rid, target_role="engineer")