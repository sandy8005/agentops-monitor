"""
Bounded autonomous job-search agent (LangGraph):
    setup -> decide -> act -> (review)* -> decide -> ... -> finalize
Ends on verified goal, accepted finish, hard limit, stalled progress, cancel, or
needed input. Guarded writes are fenced by execution generation (agent_store).
"""
import copy
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

import agent_store as store
import run_lock
from agent_controller import (ControllerDecision, ControllerOutputInvalid, allowed_actions_for,
                              llm_decide, rules_decide)
from agent_goal import AgentGoal
from agent_tools import ToolContext, ToolRejected, execute, qualified_count
from error_codes import ErrorCode, classify_exception
from logging_config import get_logger

log = get_logger(__name__)

MAX_OBSERVATIONS = 12
MAX_REJECTION_STREAK = 3
MAX_FAILURE_STREAK = 3
_LLM_OUTAGE_CODES = {ErrorCode.LLM_UNAVAILABLE, ErrorCode.LLM_RATE_LIMITED,
                     ErrorCode.LLM_QUOTA_EXHAUSTED, ErrorCode.LLM_INVALID_RESPONSE}


class StaleReviewDecision(RuntimeError):
    pass


class LoopState(TypedDict, total=False):
    run_id: int
    goal: dict
    resume_id: int
    resume_text: Optional[str]
    parsed_resume: Optional[dict]
    setup_done: bool
    iteration: int
    pending: Optional[dict]
    discovered: dict
    evaluated: dict
    searches: list
    observations: list
    review_queue: list
    human_inputs: list
    input_requests: int
    advised: list
    advice_attempted: list
    ranked: bool
    controller_mode: str
    controller_note: Optional[str]
    rejection_streak: int
    failure_streak: int
    no_progress: int
    output_failures: int
    cancel_seen: bool
    stop: Optional[dict]
    final: Optional[dict]


class DurableBudget:
    """Run-wide budget (R16). can_spend() reserves atomically right before each HTTP
    attempt; spend() is a no-op; exhausted() is a read-only gate."""

    def __init__(self, run_id, limit):
        self.run_id = run_id
        self.limit = int(limit)

    def can_spend(self):
        return store.reserve_llm_call(self.run_id, self.limit)

    def spend(self):
        pass

    def exhausted(self):
        u = store.run_usage(self.run_id)
        cap = u["llm_call_budget"] if u["llm_call_budget"] is not None else self.limit
        return u["llm_calls_reserved"] >= cap


def _ctx(state, config):
    run_id = state["run_id"]
    run_lock.check_owner(run_id)
    goal = AgentGoal(**state["goal"])
    gen = config["configurable"]["generation"]
    return run_id, goal, gen, DurableBudget(run_id, goal.limits.max_llm_calls)


def check_limits(run_id, goal, state):
    """Called before EVERY decision and EVERY tool execution (R13)."""
    if state.get("cancel_seen") or store.is_cancel_requested(run_id):
        return {"by": "user", "reason": "cancelled by user", "cancel": True}
    lim = goal.limits
    if int(state.get("iteration") or 1) > lim.max_iterations:
        return {"by": "limit", "reason": f"iteration limit reached ({lim.max_iterations})"}
    usage = store.run_usage(run_id)
    if usage["elapsed_seconds"] > lim.max_runtime_seconds:
        return {"by": "limit", "reason": f"runtime limit reached ({lim.max_runtime_seconds}s)"}
    if lim.max_cost_usd > 0 and usage["known_cost_usd"] >= lim.max_cost_usd:
        return {"by": "limit", "reason": f"estimated cost limit reached (${lim.max_cost_usd})"}
    if int(state.get("no_progress") or 0) >= lim.no_progress_limit:
        return {"by": "limit", "reason": f"no progress in {lim.no_progress_limit} consecutive "
                                         f"search/evaluation actions"}
    if int(state.get("failure_streak") or 0) >= MAX_FAILURE_STREAK:
        return {"by": "error", "reason": f"{MAX_FAILURE_STREAK} consecutive tool failures"}
    return None


def qualified_count_safe(run_id, goal):
    try:
        return len(store.qualified_job_ids(run_id, goal.qualifying_decisions))
    except Exception:
        return None


def _progress(run_id, goal, state):
    ev = state.get("evaluated") or {}
    return {"qualified": qualified_count_safe(run_id, goal),
            "target_count": goal.target_count,
            "discovered": len(state.get("discovered") or {}),
            "evaluated": sum(1 for v in ev.values() if v.get("status") == "ok"),
            "evaluation_failures": sum(1 for v in ev.values() if v.get("status") == "failed"),
            "searches": len(state.get("searches") or []),
            "iteration": int(state.get("iteration") or 1),
            "controller_mode": state.get("controller_mode"),
            "controller_note": state.get("controller_note")}


def _adapter(run_id, goal, budget, state):
    from agent_state import AgentState

    class _S(AgentState):
        def can_spend(self):
            return budget.can_spend()

        def spend(self):
            budget.spend()

        def budget_exceeded(self):
            return budget.exhausted()

    s = _S(goal=goal.description or "agent", resume_id=state.get("resume_id"),
           target_role=goal.target_role)
    s.resume_text = state.get("resume_text")
    return s


def node_setup(state: LoopState, config):
    if state.get("setup_done"):
        return {}
    run_id, goal, gen, budget = _ctx(state, config)
    from router import load_resume, do_parse_resume
    s = _adapter(run_id, goal, budget, state)
    load_resume(s, run_id)
    if not s.error:
        do_parse_resume(s, run_id)
    if s.error:
        return {"stop": {"by": "error", "reason": s.error, "setup_failed": True}}
    mode = "llm" if goal.use_llm_controller else "rules"
    return {"resume_text": s.resume_text, "parsed_resume": s.parsed_resume, "setup_done": True,
            "iteration": 1, "controller_mode": mode, "discovered": {}, "evaluated": {},
            "searches": [], "observations": [], "review_queue": [], "human_inputs": [],
            "rejection_streak": 0, "failure_streak": 0, "no_progress": 0}


def node_decide(state: LoopState, config):
    run_id, goal, gen, budget = _ctx(state, config)
    stop = check_limits(run_id, goal, state)
    if stop:
        return {"stop": stop}
    i = int(state.get("iteration") or 1)

    recorded = store.get_action(run_id, i)
    if recorded is not None:                       # replay: never re-ask the model
        return {"pending": {"action": recorded["action"], "arguments": recorded["arguments"],
                            "reason": recorded["reason"], "decided_by": recorded["decided_by"]}}

    qualified = qualified_count(ToolContext(run_id, gen, goal, state, budget, i))
    allowed = allowed_actions_for(goal, state, qualified)
    mode = state.get("controller_mode") or "rules"
    note = state.get("controller_note")
    updates = {}
    decision, decided_by = None, "rules"

    if mode == "llm":
        if budget.exhausted():
            mode, note = "rules", "controller switched to rules: LLM budget exhausted"
        else:
            from llm import create_step, finish_step, fail_step
            step_id = create_step(run_id, "agent_controller", i)
            remaining = {"iterations": goal.limits.max_iterations - i + 1,
                         "searches": goal.limits.max_searches - len(state.get("searches") or [])}
            try:
                decision = llm_decide(goal, state, qualified, allowed, remaining,
                                      run_id, step_id, budget)
                decided_by = "llm"
                finish_step(step_id, "success")
            except ControllerOutputInvalid as e:
                fail_step(step_id, e)
                streak = int(state.get("rejection_streak") or 0) + 1
                updates["rejection_streak"] = streak
                if streak >= MAX_REJECTION_STREAK:
                    mode, note = "rules", "controller switched to rules: repeated invalid output"
            except Exception as e:
                fail_step(step_id, e)
                from llm import BudgetExceeded
                code = classify_exception(e)
                if isinstance(e, BudgetExceeded) or code in _LLM_OUTAGE_CODES:
                    if goal.on_model_unavailable == "pause" and int(state.get("input_requests") or 0) < 2:
                        decision = ControllerDecision(
                            action="request_human_input",
                            arguments={"question": "The controller model is unavailable "
                                                   f"({code}). Continue with the limited "
                                                   "rules-based policy, or stop now?",
                                       "options": ["Continue with rules", "Stop now"]},
                            reason="model unavailable; the user chose to be asked")
                        decided_by = "backend"
                    mode, note = "rules", f"controller switched to rules: model unavailable ({code})"
                else:
                    raise

    if decision is None:
        decision = rules_decide(goal, state, qualified)
        decided_by = "rules"
    store.record_proposed_action(run_id, gen, i, decision.action, decision.arguments,
                                 decision.reason, decided_by)
    updates.update({"pending": {**decision.model_dump(), "decided_by": decided_by},
                    "controller_mode": mode, "controller_note": note})
    return updates


def node_act(state: LoopState, config):
    run_id, goal, gen, budget = _ctx(state, config)
    i = int(state.get("iteration") or 1)
    pending = state.get("pending") or {}
    work = copy.deepcopy(dict(state))
    stop = check_limits(run_id, goal, work)
    if stop:
        store.record_action_outcome(run_id, gen, i, "rejected",
                                    {"stopped_before_execution": stop["reason"]})
        return {"stop": stop, "pending": None}

    action, args = pending.get("action"), pending.get("arguments") or {}
    ctx = ToolContext(run_id, gen, goal, work, budget, i)
    obs_entry = {"iteration": i, "action": action, "decided_by": pending.get("decided_by")}
    try:
        obs, step_id, progressed = execute(ctx, action, args)
        store.record_action_outcome(run_id, gen, i, "executed", obs, step_id=step_id)
        obs_entry["result"] = obs
        work["failure_streak"] = 0
        if pending.get("decided_by") == "llm":
            work["rejection_streak"] = 0
        if action in ("search_jobs", "evaluate_jobs"):
            work["no_progress"] = 0 if progressed else int(work.get("no_progress") or 0) + 1
    except ToolRejected as e:
        store.record_action_outcome(run_id, gen, i, "rejected", {"rejected": str(e)})
        obs_entry["rejected"] = str(e)
        if pending.get("decided_by") == "llm":
            streak = int(work.get("rejection_streak") or 0) + 1
            work["rejection_streak"] = streak
            if streak >= MAX_REJECTION_STREAK:
                work["controller_mode"] = "rules"
                work["controller_note"] = "controller switched to rules: repeated rejected actions"
        else:
            work["no_progress"] = int(work.get("no_progress") or 0) + 1
    except (store.ExecutionLost, run_lock.ExecutionLost):
        raise
    except Exception as e:
        from sanitize import safe_exception_summary
        msg = safe_exception_summary(e)
        store.record_action_outcome(run_id, gen, i, "failed", {"failed": msg}, error=msg)
        obs_entry["failed"] = msg
        work["failure_streak"] = int(work.get("failure_streak") or 0) + 1
        if action == "rank_jobs":
            work["output_failures"] = int(work.get("output_failures") or 0) + 1

    work["observations"] = ((work.get("observations") or []) + [obs_entry])[-MAX_OBSERVATIONS:]
    work["iteration"] = i + 1
    work["pending"] = None
    try:
        store.set_controller_mode(run_id, gen, work.get("controller_mode"),
                                  _progress(run_id, goal, work))
    except store.ExecutionLost:
        raise
    except Exception:
        log.warning("progress snapshot failed", extra={"run_id": run_id})
    return work


def node_review(state: LoopState, config):
    run_id, goal, gen, budget = _ctx(state, config)
    queue = list(state.get("review_queue") or [])
    if not queue:
        return {}
    item = queue[0]
    rid = item["review_id"]
    if item.get("kind") == "input_request":
        payload = {"type": "input_request", "review_id": rid, "question": item["question"],
                   "options": item["options"]}
        kind = "input_request"
    else:
        meta = (state.get("evaluated") or {}).get(str(item["job_id"])) or {}
        from router import _step_review_reason
        payload = {"type": "review_request", "review_id": rid, "step_id": item["step_id"],
                   "job_id": item["job_id"], "job_title": meta.get("title"),
                   "company": meta.get("company"), "score": meta.get("score"),
                   "score_decision": meta.get("decision"),
                   "llm_decision": meta.get("llm_decision"),
                   "review_reason": _step_review_reason(item["step_id"])}
        kind = "job_review"
    store.create_review_request(run_id, gen, rid, kind, payload,
                                step_id=item.get("step_id"), job_id=item.get("job_id"))

    value = interrupt(payload) or {}                  # ---- PAUSE ----
    if value.get("review_id") != rid:
        raise StaleReviewDecision(f"decision for {value.get('review_id')!r} cannot answer {rid!r}")
    rec = store.review_status(rid)
    if not rec or rec["status"] not in ("submitted", "consumed"):
        raise StaleReviewDecision(f"review {rid} has no submitted decision")

    work = copy.deepcopy(dict(state))
    obs = {"iteration": int(state.get("iteration") or 1), "action": "human_" + kind,
           "review_id": rid}
    if kind == "job_review":
        decision = rec["decision"] or "Maybe"
        from router import apply_human_decision
        s = _adapter(run_id, goal, budget, state)
        apply_human_decision(s, run_id, item["step_id"], decision, rec.get("comment") or "",
                             reviewer_user_id=rec.get("reviewer_user_id"),
                             reviewer=rec.get("reviewer"))
        ev = work.setdefault("evaluated", {}).get(str(item["job_id"]))
        if ev:
            ev["final_decision"] = decision
            ev["needs_review"] = False
        work["ranked"] = False
        obs["result"] = {"job_id": item["job_id"], "human_decision": decision}
    else:
        answer = rec.get("answer")
        if answer not in item["options"]:
            raise StaleReviewDecision(f"answer {answer!r} is not one of the offered options")
        work.setdefault("human_inputs", []).append({"question": item["question"], "answer": answer})
        obs["result"] = {"answer": answer}
        if answer == "Stop now":
            work["stop"] = {"by": "user", "reason": "user chose to stop when the model was unavailable"}
    store.mark_review_consumed(rid, gen, run_id)
    work["review_queue"] = queue[1:]
    work["observations"] = ((work.get("observations") or []) + [obs])[-MAX_OBSERVATIONS:]
    return work


def node_finalize(state: LoopState, config):
    run_id, goal, gen, budget = _ctx(state, config)
    stop = state.get("stop") or {"by": "controller", "reason": "finished"}
    work = copy.deepcopy(dict(state))
    evaluated_ok = [v for v in (work.get("evaluated") or {}).values() if v.get("status") == "ok"]
    rank_error = None
    if evaluated_ok and not work.get("ranked"):
        try:
            from ranker import rank_jobs
            store.persist_rankings(run_id, gen, rank_jobs(evaluated_ok))
            work["ranked"] = True
        except (store.ExecutionLost, run_lock.ExecutionLost):
            raise
        except Exception as e:
            from sanitize import safe_exception_summary
            rank_error = safe_exception_summary(e)

    progress = _progress(run_id, goal, work)
    q = progress["qualified"] or 0
    searches = work.get("searches") or []
    all_failed = bool(searches) and all(s.get("provider_status") == "failed" for s in searches)

    if stop.get("cancel"):
        status, code = "cancelled", ErrorCode.CANCELLED
    elif stop.get("setup_failed"):
        from autonomous_graph import _stage_error_code
        status, code = "failed", _stage_error_code(stop.get("reason"))
    elif rank_error:
        status, code = "failed", ErrorCode.INTERNAL
        stop = {**stop, "reason": f"ranking could not be persisted: {rank_error}"}
    elif stop.get("by") == "error":
        status, code = ("partial_success" if q else "failed"), (None if q else ErrorCode.INTERNAL)
    elif q >= goal.target_count:
        status, code = "success", None
    elif q > 0:
        status, code = "partial_success", None
    elif all_failed:
        status, code = "failed", ErrorCode.JOB_SOURCE_UNAVAILABLE
    else:
        status, code = "no_matches", ErrorCode.NO_MATCHES
    reason = f"{stop.get('reason')} — {q}/{goal.target_count} qualified"
    if status == "success" and (work.get("output_failures") or progress["evaluation_failures"]):
        reason += " (some evaluations or advice failed; see trace)"
    work["final"] = {"status": status, "reason": reason,
                     "error_code": code.value if code else None, "progress": progress}
    return work


def route_after_setup(state):
    return "finalize" if state.get("stop") else "decide"


def route_after_decide(state):
    return "finalize" if state.get("stop") else "act"


def route_after_act(state):
    stop = state.get("stop") or {}
    if stop.get("cancel"):
        return "finalize"              # never pause for a human after a cancel (R13)
    if state.get("review_queue"):
        return "review"
    if stop:
        return "finalize"
    return "decide"


def route_after_review(state):
    if state.get("review_queue"):
        return "review"
    return "finalize" if state.get("stop") else "decide"


def build_agent_graph(checkpointer=None):
    g = StateGraph(LoopState)
    g.add_node("setup", node_setup)
    g.add_node("decide", node_decide)
    g.add_node("act", node_act)
    g.add_node("review", node_review)
    g.add_node("finalize", node_finalize)
    g.set_entry_point("setup")
    g.add_conditional_edges("setup", route_after_setup, {"decide": "decide", "finalize": "finalize"})
    g.add_conditional_edges("decide", route_after_decide, {"act": "act", "finalize": "finalize"})
    g.add_conditional_edges("act", route_after_act,
                            {"review": "review", "decide": "decide", "finalize": "finalize"})
    g.add_conditional_edges("review", route_after_review,
                            {"review": "review", "decide": "decide", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def _recursion_limit(goal: AgentGoal):
    return goal.limits.max_iterations * 3 + 40


def pending_interrupt_value(graph, config):
    snap = graph.get_state(config)
    intr = getattr(snap, "interrupts", None) or ()
    if not intr:
        for t in getattr(snap, "tasks", ()) or ():
            if getattr(t, "interrupts", None):
                intr = t.interrupts
                break
    if not intr:
        return None
    v = intr[0].value
    return v if isinstance(v, dict) else {"payload": v}


def _handle_result(run_id, gen, result):
    if isinstance(result, dict) and result.get("__interrupt__"):
        v = result["__interrupt__"][0].value
        payload = v if isinstance(v, dict) else {"payload": v}
        store.mark_waiting(run_id, gen, payload)
        log.info("agent run paused (%s)", payload.get("type"), extra={"run_id": run_id})
        return result
    final = (result or {}).get("final") or {"status": "failed", "reason": "no final state",
                                            "error_code": ErrorCode.INTERNAL.value, "progress": {}}
    store.finalize(run_id, gen, final["status"], final["reason"], final["error_code"],
                   final["progress"])
    log.info("agent run finished: %s (%s)", final["status"], final["reason"], extra={"run_id": run_id})
    return result


def _fail(run_id, gen, e):
    if isinstance(e, (store.ExecutionLost, run_lock.ExecutionLost)) or run_lock.is_lost(run_id):
        log.warning("agent execution abandoned: %s", e, extra={"run_id": run_id})
        return {"abandoned": True}
    from sanitize import safe_exception_summary
    try:
        store.finalize(run_id, gen, "failed", safe_exception_summary(e), classify_exception(e), {})
    except Exception as fin:
        log.error("could not finalize failed agent run: %s", fin, extra={"run_id": run_id})
    log.error("agent run failed: %s", safe_exception_summary(e), extra={"run_id": run_id})
    return {"error": True}


def run_agent_loop(run_id, queue_attempt=1, checkpointer_factory=None):
    from checkpointing import open_checkpointer
    cfg = store.load_run_config(run_id)
    if not cfg or not cfg["goal"]:
        raise ValueError(f"run {run_id} has no agent goal")
    goal = AgentGoal(**cfg["goal"])
    gen = store.begin_execution(run_id, new_attempt=True)
    if cfg["cancel_requested"]:
        store.finalize(run_id, gen, "cancelled", "cancelled before start", ErrorCode.CANCELLED, {})
        return {"cancelled": True}
    initial = {"run_id": run_id, "goal": goal.model_dump(), "resume_id": cfg["resume_id"]}
    try:
        with (checkpointer_factory or open_checkpointer)() as cp:
            graph = build_agent_graph(cp)
            config = {"configurable": {"thread_id": str(run_id), "generation": gen},
                      "recursion_limit": _recursion_limit(goal)}
            snap = graph.get_state(config)
            # A retried attempt CONTINUES from the last checkpoint (replay-safely).
            if snap and snap.next and not pending_interrupt_value(graph, config):
                result = graph.invoke(None, config=config)
            else:
                result = graph.invoke(initial, config=config)
            return _handle_result(run_id, gen, result)
    except Exception as e:
        return _fail(run_id, gen, e)


def resume_agent_loop(run_id, payload, queue_attempt=1, checkpointer_factory=None):
    """Decision is bound to payload['review_id']; a mismatch is stale (R01)."""
    from checkpointing import open_checkpointer
    cfg = store.load_run_config(run_id)
    goal = AgentGoal(**cfg["goal"])
    gen = store.begin_execution(run_id, new_attempt=(queue_attempt or 1) > 1)
    try:
        with (checkpointer_factory or open_checkpointer)() as cp:
            graph = build_agent_graph(cp)
            config = {"configurable": {"thread_id": str(run_id), "generation": gen},
                      "recursion_limit": _recursion_limit(goal)}
            pending = pending_interrupt_value(graph, config)

            if cfg["cancel_requested"]:
                values = graph.get_state(config).values or {}
                ok = [v for v in (values.get("evaluated") or {}).values() if v.get("status") == "ok"]
                if ok:
                    try:
                        from ranker import rank_jobs
                        store.persist_rankings(run_id, gen, rank_jobs(ok))
                    except Exception as e:
                        log.warning("ranking on cancel failed: %s", type(e).__name__,
                                    extra={"run_id": run_id})
                store.finalize(run_id, gen, "cancelled", "cancelled by user",
                               ErrorCode.CANCELLED, {})
                return {"cancelled": True}

            rid = (payload or {}).get("review_id")
            if not pending or pending.get("review_id") != rid:
                if pending:
                    store.mark_waiting(run_id, gen, pending)
                log.warning("stale resume for review %s ignored (checkpoint waits on %s)",
                            rid, (pending or {}).get("review_id"), extra={"run_id": run_id})
                return {"stale": True}
            result = graph.invoke(Command(resume={"review_id": rid}), config=config)
            return _handle_result(run_id, gen, result)
    except Exception as e:
        return _fail(run_id, gen, e)