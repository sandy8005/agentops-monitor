"""
Prove the LangGraph interrupt/resume cycle WITHOUT needing the LLM to produce a
real disagreement. We build a minimal 2-node graph: a node that flags a job and
routes to human_review (which interrupts), then verify:
  1. invoke() PAUSES (returns __interrupt__), and
  2. Command(resume=...) CONTINUES and completes.

This isolates the interrupt mechanism from scoring/quota entirely.
Run:  python test_interrupt_cycle.py
"""
import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres import PostgresSaver


def _db_uri():
    return (f"host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')} "
            f"dbname={os.getenv('DB_NAME')} user={os.getenv('DB_USER')} "
            f"password={os.getenv('DB_PASSWORD')}")


class S(TypedDict, total=False):
    flagged: bool
    human_decision: Optional[str]
    done: bool


def node_work(state: S) -> dict:
    # pretend this job got flagged for review
    return {"flagged": True}

def node_review(state: S) -> dict:
    human = interrupt({"type": "review_request", "reason": "forced-for-test"})
    return {"human_decision": (human or {}).get("decision", "Maybe"),
            "flagged": False, "done": True}

def route(state: S) -> str:
    if state.get("flagged"):
        return "review"
    return "end"


def build():
    g = StateGraph(S)
    g.add_node("work", node_work)
    g.add_node("review", node_review)
    g.set_entry_point("work")
    g.add_conditional_edges("work", route, {"review": "review", "end": END})
    g.add_edge("review", END)
    return g


def main():
    from dotenv import load_dotenv
    load_dotenv()
    thread = "interrupt-test-1"
    config = {"configurable": {"thread_id": thread}}

    with PostgresSaver.from_conn_string(_db_uri()) as cp:
        cp.setup()
        graph = build().compile(checkpointer=cp)

        # 1. First invoke — should PAUSE at the interrupt
        result = graph.invoke({}, config=config)
        assert "__interrupt__" in result, f"expected pause, got keys: {list(result.keys())}"
        intr = result["__interrupt__"]
        print("PASS: graph PAUSED at interrupt ->", intr[0].value)

        # 2. Resume with a decision — should CONTINUE and finish
        final = graph.invoke(Command(resume={"decision": "Apply"}), config=config)
        assert final.get("done") is True, f"expected done, got: {final}"
        assert final.get("human_decision") == "Apply", final
        print("PASS: graph RESUMED with decision ->", final["human_decision"])

    print("\nInterrupt/resume cycle WORKS: pause -> checkpoint -> resume -> continue.")


if __name__ == "__main__":
    main()