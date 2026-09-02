"""
The autonomous agent driver: plan → act → repeat until the planner says stop.
This loop IS the agent. All intelligence lives in the planner (what to do next,
0 LLM calls) and the tools (which spend Gemini only where judgment is needed).
The loop itself is deliberately tiny — that's the sign the design is right.
"""
from agent_state import AgentState
from planner import plan_next_action
from router import dispatch
from llm import create_run, finish_run

TERMINAL = {"done", "fail", "finish_no_matches", "finish_budget","finish_cancelled"}


def run_agent_autonomous(resume_id, target_role=None, location=None,
                         work_mode=None, employment_type=None,
                         evaluate=False, run_id=None, max_llm_calls=30,
                         max_steps=200):
    """
    Drive the autonomous loop. Returns the final AgentState.
    max_steps is a hard safety bound so a planner bug can never loop forever (#20).
    """
    state = AgentState(
        goal="match resume to jobs", resume_id=resume_id,
        target_role=target_role, location=location,
        work_mode=work_mode, employment_type=employment_type, evaluate=evaluate
    )
    state.max_llm_calls = max_llm_calls

    if run_id is None:
        run_id = create_run("autonomous job search", resume_id=resume_id,
                            target_role=target_role, location=location,
                            work_mode=work_mode, employment_type=employment_type)

    steps_taken = 0
    final_status = "success"

    try:
        while not state.done:
            # hard safety bound — a planner that never returns a terminal action
            # still can't loop forever.
            if steps_taken >= max_steps:
                state.error = f"max_steps ({max_steps}) exceeded — stopping"
                final_status = "failed"
                break
            steps_taken += 1

            action = plan_next_action(state)   # planner decides — 0 LLM calls

            if action in TERMINAL:
                if action == "fail":
                    final_status = "failed"
                elif action == "finish_no_matches":
                    final_status = "no_matches"
                elif action == "finish_cancelled":
                    final_status = "cancelled"
                elif action == "finish_budget":
                    final_status = "completed_with_errors"
                    print(f"    ⚠ LLM call budget ({state.max_llm_calls}) reached — stopping early")
                else:  # "done"
                    final_status = ("completed_with_errors"
                                    if state.failed_jobs > 0 else "success")
                # 'done' → success
                state.done = True
                break

            dispatch(action, state, run_id)    # router executes the tool

            # if a tool set an error, the planner will route to 'fail' next loop
        # end while

        if state.error and final_status == "success":
            final_status = "completed_with_errors"

    finally:
        finish_run(run_id, final_status)

    print(f"Run {run_id} finished: {final_status} "
          f"({state.llm_calls_made} LLM calls, {steps_taken} steps)")
    if state.ranked:
        print("\nRANKED JOBS:")
        for i, r in enumerate(state.ranked, 1):
            print(f"{i}. {r['title']} ({r['company']}) — "
                  f"score {r['score']} ({r['decision']}), judge: {r['llm_decision']}")

    return state


if __name__ == "__main__":
    import sys
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_agent_autonomous(resume_id=rid, target_role="engineer")