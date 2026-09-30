"""
The controller's toolset: search_jobs, evaluate_jobs, rank_jobs, generate_advice,
request_human_input, finish.

Contract for every tool:
  * Arguments are validated with a strict Pydantic model (extra fields rejected).
  * Semantic checks run against BACKEND state, not the model's claims: job ids must
    be ids this run discovered; queries cannot change fixed constraints; searches
    cannot repeat; limits are checked before execution.
  * A rejected call returns an observation explaining why and has NO side effects.
  * Execution is REPLAY-SAFE: re-running the same iteration after a crash reuses
    persisted effects (successful searches, existing job evaluations, idempotent
    ranking/advice writes) instead of duplicating them — but a search that FAILED
    transiently in an earlier worker generation is genuinely retried (plan_search).
  * Observations are compact counts and ids. Job text is untrusted data and never
    becomes an instruction; titles are only shown to the controller fenced and
    truncated.
"""
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

import agent_store as store
from agent_goal import AgentGoal, seniority_conflict, validate_search_query
from logging_config import get_logger

log = get_logger(__name__)

TOOL_NAMES = ("search_jobs", "evaluate_jobs", "rank_jobs", "generate_advice",
              "request_human_input", "finish")
LIVE_PROVIDERS = {"adzuna", "remotive"}
MAX_INPUT_REQUESTS = 2


class ToolRejected(Exception):
    """The requested action is not allowed. No side effects happened."""


# ------------------------------------------------------------------ arguments --

class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["adzuna", "remotive", "pool"]
    query: str = Field(..., min_length=2, max_length=80)


class EvaluateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_ids: List[int] = Field(..., min_length=1, max_length=10)

    @field_validator("job_ids")
    @classmethod
    def _unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("job_ids must be unique")
        return v


class RankArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdviceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_ids: List[int] = Field(..., min_length=1, max_length=3)

    @field_validator("job_ids")
    @classmethod
    def _unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("job_ids must be unique")
        return v


MAX_ADVICE_JOBS = 3     # run-wide cap on jobs that receive advice (enforced HERE)


class HumanInputArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., min_length=5, max_length=300)
    options: List[str] = Field(..., min_length=2, max_length=4)

    @field_validator("options")
    @classmethod
    def _short(cls, v):
        v = [" ".join(o.split()) for o in v]
        if any(not o or len(o) > 60 for o in v) or len(set(v)) != len(v):
            raise ValueError("options must be 2-4 distinct labels of at most 60 characters")
        return v


class FinishArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(..., min_length=3, max_length=300)


ARG_MODELS = {"search_jobs": SearchArgs, "evaluate_jobs": EvaluateArgs,
              "rank_jobs": RankArgs, "generate_advice": AdviceArgs,
              "request_human_input": HumanInputArgs, "finish": FinishArgs}

TOOL_SPECS = {
    "search_jobs": {"arguments": {"provider": "one of the allowed providers",
                                  "query": "a plain job title (2-80 chars)"},
                    "purpose": "Search one provider. Location, work mode, employment type "
                               "and seniority are applied automatically and cannot be changed."},
    "evaluate_jobs": {"arguments": {"job_ids": "1-N ids from 'unevaluated_eligible_job_ids'"},
                      "purpose": "Score discovered jobs against the resume using the fixed scoring rules."},
    "rank_jobs": {"arguments": {}, "purpose": "Rank and persist all evaluated jobs."},
    "generate_advice": {"arguments": {"job_ids": "1-3 ids of qualifying evaluated jobs"},
                        "purpose": "Evidence-checked resume suggestions for top matches."},
    "request_human_input": {"arguments": {"question": "short question", "options": ["2-4 labels"]},
                            "purpose": "Pause for a necessary user decision. Use sparingly."},
    "finish": {"arguments": {"reason": "why you are stopping"},
               "purpose": "Propose completion. The backend verifies progress from persisted results."},
}


def validate_arguments(action, arguments):
    """Parse arguments with the tool's model. Raises ToolRejected on invalid input."""
    if action not in ARG_MODELS:
        raise ToolRejected(f"unknown action {action!r}; allowed: {list(TOOL_NAMES)}")
    try:
        return ARG_MODELS[action](**(arguments or {}))
    except ValidationError as e:
        msgs = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                         for err in e.errors()[:4])
        raise ToolRejected(f"invalid arguments for {action}: {msgs}")
    except TypeError as e:
        raise ToolRejected(f"invalid arguments for {action}: {e}")


# ------------------------------------------------------------------- helpers --

class ToolContext:
    """What a tool needs: run identity, the fenced generation, the goal, the loop
    state (mutated in place), the durable budget and the iteration number."""

    def __init__(self, run_id, generation, goal: AgentGoal, state, budget, iteration,
                 limit_check=None):
        self.run_id = run_id
        self.generation = generation
        self.goal = goal
        self.state = state
        self.budget = budget
        self.iteration = iteration
        # Callable -> None | stop dict. Multi-unit tools call it between units so
        # cancellation and runtime/cost limits are honoured inside a tool (N06).
        self.limit_check = limit_check or (lambda: None)


def unevaluated_eligible_ids(state):
    ev = state.get("evaluated") or {}
    return [int(k) for k, v in (state.get("discovered") or {}).items()
            if v.get("eligible") and k not in ev]


def qualified_count(ctx):
    return len(store.qualified_job_ids(ctx.run_id, ctx.goal.qualifying_decisions,
                                       verified_only=ctx.goal.require_verified_matches))


def _tool_state(ctx):
    """An AgentState whose budget methods delegate to the DURABLE run budget, so the
    existing, tested router.do_process_job can be reused unchanged."""
    from agent_state import AgentState
    g = ctx.goal

    class _ToolState(AgentState):
        def can_spend(self):
            return ctx.budget.can_spend()

        def spend(self):
            ctx.budget.spend()

        def budget_exceeded(self):
            return ctx.budget.exhausted()

    ts = _ToolState(goal=g.description or "agent", resume_id=ctx.state.get("resume_id"),
                    target_role=g.target_role, location=g.constraints.location or None,
                    work_mode=g.constraints.work_mode or None,
                    employment_type=g.constraints.employment_type or None,
                    evaluate=g.evaluate_quality, live_only=True)
    ts.resume_text = ctx.state.get("resume_text")
    ts.parsed_resume = ctx.state.get("parsed_resume")
    ts.model_policy = g.model_policy
    return ts


# ------------------------------------------------------------------- search --

def _collect_candidates(ctx, provider, query):
    from job_source import search_jobs as pool_search
    c = ctx.goal.constraints
    jobs = pool_search(query, c.location or None, c.work_mode or None,
                       c.employment_type or None, live_only=(provider in LIVE_PROVIDERS),
                       run_id=ctx.run_id if provider in LIVE_PROVIDERS else None)
    if provider in LIVE_PROVIDERS:
        jobs = [j for j in jobs if (j.get("source") or "").lower() == provider]
    else:
        jobs = [j for j in jobs if (j.get("source") or "").lower() not in LIVE_PROVIDERS]
    # Deterministic, bounded selection (R14): newest ids first, capped.
    jobs = sorted(jobs, key=lambda j: j["id"], reverse=True)[:ctx.goal.limits.max_jobs_per_search]
    return jobs


# Provider outcomes that can change on a retry (a later worker generation should
# really call the provider again) vs. ones that cannot (credentials/config, a body
# the provider will keep sending malformed). Unknown/legacy details count as
# transient: retrying costs one request, wrongly never retrying loses the search.
NON_TRANSIENT_PROVIDER_FAILURES = frozenset({"missing_keys", "auth_error", "invalid_response"})


def plan_search(history, iteration, generation):
    """Decide how THIS execution of search_jobs(provider, query) treats earlier
    recorded executions of the same (provider, query) in this run.

    Returns {"mode": "fetch" | "reuse" | "reject", "prior": row | None,
             "retrying_generation": int | None}

      * no history                                   -> fetch
      * recorded by a DIFFERENT iteration (another
        decision already searched this)              -> reject (no duplicate searches)
      * same decision, any SUCCESSFUL execution      -> reuse (postings are persisted)
      * same decision, same generation (LangGraph
        replay inside one execution)                 -> reuse (never refetch)
      * same decision, only FAILED executions in
        EARLIER generations:
            transient failure (429/5xx/network/...)  -> fetch  (a real provider retry)
            non-transient (auth/keys/bad body)       -> reuse
    """
    if not history:
        return {"mode": "fetch", "prior": None, "retrying_generation": None}
    if any(h.get("iteration") != iteration for h in history):
        return {"mode": "reject", "prior": history[0], "retrying_generation": None}
    ok = [h for h in history if h.get("provider_status") == "success"]
    if ok:
        return {"mode": "reuse", "prior": ok[0], "retrying_generation": None}
    same = [h for h in history if h.get("execution_generation") == generation]
    if same:
        return {"mode": "reuse", "prior": same[0], "retrying_generation": None}
    latest = history[0]
    if (latest.get("provider_detail") or "") in NON_TRANSIENT_PROVIDER_FAILURES:
        return {"mode": "reuse", "prior": latest, "retrying_generation": None}
    return {"mode": "fetch", "prior": latest,
            "retrying_generation": latest.get("execution_generation")}


def tool_search_jobs(ctx, args: SearchArgs):
    goal, state = ctx.goal, ctx.state
    if args.provider not in goal.providers:
        raise ToolRejected(f"provider {args.provider!r} is not enabled for this run "
                           f"(allowed: {goal.providers})")
    ok, qnorm, why = validate_search_query(args.query, goal)
    if not ok:
        raise ToolRejected(why)
    plan = plan_search(store.search_history(ctx.run_id, args.provider, qnorm),
                       ctx.iteration, ctx.generation)
    if plan["mode"] == "reject":
        raise ToolRejected(f"'{qnorm}' was already searched on {args.provider}; "
                           f"choose a different title or provider")
    done = [s for s in state.get("searches") or [] if s.get("iteration") != ctx.iteration]
    if len(done) >= goal.limits.max_searches:
        raise ToolRejected(f"search limit reached ({goal.limits.max_searches})")

    from llm import create_step, finish_step, fail_step
    step_id = create_step(ctx.run_id, f"agent_search:{args.provider}", ctx.iteration)
    try:
        prior = plan.get("prior")
        if plan["mode"] == "reuse":
            # Same decision, result already known and NOT worth refetching: a
            # success (postings are persisted), a same-generation replay, or a
            # non-transient failure (bad credentials won't fix themselves).
            provider_status = prior.get("provider_detail") or (
                "success" if prior["provider_status"] == "success" else "failed")
        elif args.provider == "adzuna":
            from adzuna_jobs import fetch_and_upsert_adzuna
            _, _, provider_status = fetch_and_upsert_adzuna(
                args.query, goal.constraints.location or None,
                limit=goal.limits.max_jobs_per_search, run_id=ctx.run_id, step_id=step_id)
        elif args.provider == "remotive":
            from live_jobs import fetch_and_upsert_remotive
            _, _, provider_status = fetch_and_upsert_remotive(
                args.query, goal.constraints.location or None,
                limit=goal.limits.max_jobs_per_search, run_id=ctx.run_id, step_id=step_id)
        else:
            provider_status = "success"
        fetched = plan["mode"] == "fetch"
        reused_from = None if fetched else prior.get("execution_generation")

        succeeded = provider_status in ("success", "empty")
        candidates = _collect_candidates(ctx, args.provider, args.query) if succeeded else []
        discovered = state.setdefault("discovered", {})
        new = dup = eligible = 0
        rejections = {}
        for j in candidates:
            key = str(j["id"])
            if key in discovered:
                dup += 1
                continue
            new += 1
            reason = None
            conflict = seniority_conflict(j.get("title"), goal.constraints.seniority)
            if conflict:
                reason = "seniority"
            elif j.get("geo_eligibility") == "ineligible":
                reason = "location"
            if reason:
                rejections[reason] = rejections.get(reason, 0) + 1
            else:
                eligible += 1
            discovered[key] = {"id": j["id"], "title": (j.get("title") or "")[:120],
                               "company": (j.get("company") or "")[:80],
                               "source": j.get("source"), "apply_url": j.get("apply_url"),
                               "eligible": reason is None, "reject_reason": reason,
                               "found_by": f"{args.provider}:{qnorm}"}
        obs = {
            "provider": args.provider, "query": qnorm,
            "provider_status": "success" if succeeded else "failed",
            "provider_detail": provider_status,
            "new_jobs": new, "duplicates": dup, "eligible_jobs": eligible,
            "rejection_summary": rejections,
            "fetched": fetched,
        }
        if plan["mode"] == "fetch" and plan.get("retrying_generation") is not None:
            obs["retry_of_generation"] = plan["retrying_generation"]
        if reused_from is not None:
            obs["reused_from_generation"] = reused_from
        if not succeeded:
            obs["note"] = ("search FAILED — this is not evidence that no jobs exist; "
                           "try another provider or title")
        store.record_search(ctx.run_id, ctx.generation, ctx.iteration, args.provider,
                            qnorm, goal.constraints.location, obs, fetched=fetched,
                            reused_from_generation=reused_from)
        state.setdefault("searches", []).append({**obs, "iteration": ctx.iteration})
        finish_step(step_id, "success" if succeeded else "failed")
        return obs, step_id, (eligible > 0)
    except Exception as e:
        fail_step(step_id, e)
        raise


# ----------------------------------------------------------------- evaluate --

def _existing_evaluation(run_id, job_id):
    """A SUCCESSFUL, already-persisted evaluation of this job in this run (replay
    safety: evaluating twice would create duplicate steps and LLM calls)."""
    from database import get_connection
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, match_score, score_decision, llm_decision, final_decision,
                   needs_human_review, review_status, step_name
            FROM steps
            WHERE run_id = %s AND status = 'success' AND retrieved_context ? 'job_id'
              AND retrieved_context->>'job_id' = %s
            ORDER BY id DESC LIMIT 1
        """, (run_id, str(job_id)))
        r = cur.fetchone()
    if not r:
        return None
    return {"step_id": r[0], "score": float(r[1]) if r[1] is not None else None,
            "decision": r[2], "llm_decision": r[3], "final_decision": r[4],
            "needs_review": bool(r[5]) and r[6] is None}


def tool_evaluate_jobs(ctx, args: EvaluateArgs):
    goal, state = ctx.goal, ctx.state
    if len(args.job_ids) > goal.limits.max_evaluate_batch:
        raise ToolRejected(f"at most {goal.limits.max_evaluate_batch} jobs per evaluation")
    allowed = set(unevaluated_eligible_ids(state))
    invalid = [i for i in args.job_ids if i not in allowed]
    if invalid:
        raise ToolRejected(f"job ids {invalid[:5]} are not unevaluated eligible jobs discovered "
                           f"by this run; use ids from 'unevaluated_eligible_job_ids'")

    from router import do_process_job
    postings = store.load_postings(args.job_ids)
    evaluated = state.setdefault("evaluated", {})
    counts = {"Apply": 0, "Maybe": 0, "Skip": 0}
    failed, flagged, step_ids = [], [], []
    for job_id in args.job_ids:
        stop = ctx.limit_check()                           # N06: between work units
        if stop:
            if stop.get("cancel"):
                state["cancel_seen"] = True
            state["limit_stop"] = stop
            break
        meta = state["discovered"][str(job_id)]
        prior = _existing_evaluation(ctx.run_id, job_id)
        if prior is not None:
            result = {**prior, "job_id": job_id, "title": meta["title"],
                      "company": meta["company"], "apply_url": meta.get("apply_url")}
        else:
            job = postings.get(job_id)
            if job is None:
                failed.append(job_id)
                evaluated[str(job_id)] = {"job_id": job_id, "status": "failed",
                                          "error": "posting no longer exists"}
                continue
            ts = _tool_state(ctx)
            ts.jobs = [job]
            ts.current_job_index = 0
            do_process_job(ts, ctx.run_id)
            if ts.cancelled:
                state["cancel_seen"] = True
                break
            if ts.failed_jobs or not ts.job_results:
                failed.append(job_id)
                evaluated[str(job_id)] = {"job_id": job_id, "status": "failed",
                                          "error": "evaluation failed (see trace)"}
                continue
            result = dict(ts.job_results[-1])
            result["job_id"] = job_id
        result["status"] = "ok"
        evaluated[str(job_id)] = result
        step_ids.append(result.get("step_id"))
        fd = result.get("final_decision") or result.get("decision")
        if fd in counts:
            counts[fd] += 1
        if result.get("needs_review"):
            flagged.append(job_id)
            rid = f"{ctx.run_id}:job:{result['step_id']}"
            queue = state.setdefault("review_queue", [])
            if not any(q["review_id"] == rid for q in queue):
                queue.append({"review_id": rid, "step_id": result["step_id"], "job_id": job_id})
    state["ranked"] = False          # new evaluations invalidate the last ranking
    obs = {"evaluated": len(step_ids), "decisions": counts, "failed_job_ids": failed,
           "needs_human_review": flagged, "qualified_so_far": qualified_count(ctx),
           "target_count": goal.target_count,
           "unevaluated_eligible_remaining": len(unevaluated_eligible_ids(state))}
    if state.get("cancel_seen"):
        obs["note"] = "cancellation requested — evaluation stopped"
    progress = counts["Apply"] + (counts["Maybe"] if "Maybe" in goal.qualifying_decisions else 0)
    return obs, None, progress > 0


# --------------------------------------------------------------------- rank --

def tool_rank_jobs(ctx, args: RankArgs):
    from ranker import rank_jobs
    ok = [v for v in (ctx.state.get("evaluated") or {}).values() if v.get("status") == "ok"]
    if not ok:
        raise ToolRejected("nothing has been evaluated yet")
    ranked = rank_jobs(ok)
    store.persist_rankings(ctx.run_id, ctx.generation, ranked)     # raises on failure (R03)
    ctx.state["ranked"] = True
    top = [{"job_id": r["job_id"], "final_decision": r.get("final_decision"),
            "score": r.get("score")} for r in ranked[:5]]
    return {"ranked": len(ranked), "top": top, "persisted": True}, None, False


# ------------------------------------------------------------------- advice --

def tool_generate_advice(ctx, args: AdviceArgs):
    from router import _reqs_cache_get, _reqs_cache_key, resume_content_hash
    from scorer import calculate_match_score
    from resume_advisor import build_suggestions, advice_summary
    from llm import create_step, finish_step, fail_step
    goal, state = ctx.goal, ctx.state
    evaluated = state.get("evaluated") or {}
    allowed = {"Apply", "Maybe"}
    bad = [i for i in args.job_ids
           if str(i) not in evaluated or evaluated[str(i)].get("status") != "ok"
           or (evaluated[str(i)].get("final_decision") not in allowed)]
    if bad:
        raise ToolRejected(f"job ids {bad} are not evaluated Apply/Maybe matches of this run")
    # N10: repeats and the run-wide cap are enforced HERE, against durable state
    # (persisted advice) as well as the checkpointed attempt list — not only by the
    # rules policy. A replayed iteration may re-request exactly the jobs it already
    # persisted; those are reused, never regenerated.
    persisted = set(store.advised_job_ids(ctx.run_id))
    attempted = set(state.get("advice_attempted") or []) | persisted
    replay = bool(set(args.job_ids) <= persisted)
    repeats = [i for i in args.job_ids if i in attempted]
    if repeats and not replay:
        raise ToolRejected(f"advice was already generated or attempted for {repeats}")
    if not replay and len(attempted | set(args.job_ids)) > MAX_ADVICE_JOBS:
        raise ToolRejected(f"advice is limited to {MAX_ADVICE_JOBS} jobs per run "
                           f"({len(attempted)} already used)")
    # Recorded up front so a failing advice call is never retried in a loop.
    state["advice_attempted"] = sorted(attempted | set(args.job_ids))
    postings = store.load_postings(args.job_ids)
    step_id = create_step(ctx.run_id, "agent_generate_advice", ctx.iteration)
    done, failed, notes = [], [], {}
    resume_hash = resume_content_hash(state.get("resume_text") or "")
    try:
        for job_id in args.job_ids:
            if job_id in persisted:
                done.append(job_id)          # already persisted: reuse, don't regenerate
                notes[str(job_id)] = "reused"
                continue
            if ctx.limit_check():
                break
            job = postings.get(job_id)
            if job is None:
                failed.append(job_id)
                continue
            try:
                reqs, _prov = _reqs_cache_get(_reqs_cache_key(job["title"], job["description"]))
                reqs = reqs or {"required_skills": [], "required_any_of": [],
                                "preferred_skills": [], "min_years_experience": 0,
                                "responsibilities": []}
                from skills import normalize_requirements
                reqs = normalize_requirements(reqs)
                sc = calculate_match_score(state.get("parsed_resume"), reqs,
                                           state.get("resume_text"), job, None)
                suggestions, note = build_suggestions(
                    state.get("parsed_resume"), state.get("resume_text"), job, reqs, sc,
                    use_llm=goal.use_llm_advice, run_id=ctx.run_id, step_id=step_id,
                    budget=ctx.budget)
                store.persist_advice(ctx.run_id, ctx.generation, job_id, job["title"],
                                     advice_summary(job, suggestions, note), suggestions,
                                     state.get("resume_id"), resume_hash)
                done.append(job_id)
                notes[str(job_id)] = note
            except store.ExecutionLost:
                raise
            except Exception as e:
                log.warning("advice failed for job %s: %s", job_id, type(e).__name__,
                            extra={"run_id": ctx.run_id, "step_id": step_id})
                failed.append(job_id)
        advised = set(state.get("advised") or [])
        state["advised"] = sorted(advised | set(done))
        finish_step(step_id, "success" if not failed else "failed")
    except Exception as e:
        fail_step(step_id, e)
        raise
    if failed:
        state["output_failures"] = int(state.get("output_failures") or 0) + len(failed)
    return {"advised": done, "failed_job_ids": failed, "generation": notes}, step_id, False


# ------------------------------------------------------------- human input --

def tool_request_human_input(ctx, args: HumanInputArgs):
    asked = int(ctx.state.get("input_requests") or 0)
    if asked >= MAX_INPUT_REQUESTS:
        raise ToolRejected("the input-request limit for this run is reached; decide with the "
                           "information available or finish")
    rid = f"{ctx.run_id}:input:{ctx.iteration}"
    queue = ctx.state.setdefault("review_queue", [])
    if not any(q["review_id"] == rid for q in queue):
        queue.append({"review_id": rid, "kind": "input_request", "question": args.question,
                      "options": args.options})
        ctx.state["input_requests"] = asked + 1
    return {"paused_for_input": True, "review_id": rid}, None, False


# ------------------------------------------------------------------- finish --

def tool_finish(ctx, args: FinishArgs):
    remaining = unevaluated_eligible_ids(ctx.state)
    qualified = qualified_count(ctx)
    if qualified < ctx.goal.target_count and remaining:
        raise ToolRejected(f"goal not met ({qualified}/{ctx.goal.target_count}) and "
                           f"{len(remaining)} eligible discovered jobs are unevaluated")
    ctx.state["stop"] = {"by": "controller", "reason": args.reason}
    return {"finish_accepted": True, "qualified": qualified,
            "target_count": ctx.goal.target_count}, None, False


EXECUTORS = {"search_jobs": tool_search_jobs, "evaluate_jobs": tool_evaluate_jobs,
             "rank_jobs": tool_rank_jobs, "generate_advice": tool_generate_advice,
             "request_human_input": tool_request_human_input, "finish": tool_finish}


def execute(ctx, action, arguments):
    """Validate then run. Returns (observation, step_id, made_progress). Raises
    ToolRejected (no side effects) or the tool's own exception (failure)."""
    args = validate_arguments(action, arguments)
    return EXECUTORS[action](ctx, args)