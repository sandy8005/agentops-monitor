# DESIGN: Human-in-the-Loop Review via LangGraph Interrupts

Status: **Designed, not yet implemented.** This document is the spec to build from.

## Motivation

The agent already flags risky decisions (score↔LLM disagreement, hallucination,
low evaluation, insufficient requirements, experience discrepancy) and a human
reviews them **after** the run completes, in the dashboard queue. That is
*post-hoc* review — the run is already finished when the human weighs in.

This feature makes review **inline**: when a job is flagged mid-run, the agent
*pauses*, waits for the human's decision, and *resumes based on it*. The human is
in the loop, not after it. The human's Apply/Maybe/Skip choice becomes the
authoritative decision for that job; the agent's original score/LLM decisions are
preserved for the audit trail.

This is built on LangGraph's `interrupt()` + checkpointing, which lets a graph
suspend mid-execution, persist its full state, and resume later from the exact
node that paused — even across separate HTTP requests.

## Control flow

```
process_job node
        │
        ▼
   needs_review?  ──no──▶ next job / rank
        │yes
        ▼
route to human_review node
        │
        ▼
   interrupt(payload)   ──▶ [checkpointer AUTO-SAVES full graph state]
        │
        ▼
   invoke() returns to the FastAPI wrapper
        │
        ▼
   wrapper sets run status = 'waiting_for_human'
        │   [background task ends — run is SUSPENDED in the DB]
        ▼
Dashboard polls, shows the paused review (from the interrupt payload)
        │
        ▼
Human submits Apply / Maybe / Skip + comment
        │
        ▼
POST /runs/{id}/resume ──▶ graph.invoke(Command(resume=decision), thread_id=run_id)
        │
        ▼
LangGraph loads checkpoint; human_review's interrupt() RETURNS the decision
        │
        ▼
apply human decision (authoritative; keep agent's original for audit)
        │
        ▼
route back into the loop ──more flagged jobs?──▶ (pause again)
        │ no more
        ▼
rank ──▶ advice ──▶ finish run (success / completed_with_errors)
```

**Multi-flag runs:** with N flagged jobs, the pause/resume cycle happens N times.
The run status oscillates `running → waiting_for_human → running → … → success`.

## The four components

### 1. Checkpointer (persistence)
LangGraph needs a persistence backend to store paused state. Use
`langgraph.checkpoint.postgres.PostgresSaver` against the existing Postgres
(`langgraph-checkpoint` is already an installed dependency). It manages its own
tables (`checkpoints`, `checkpoint_writes`, …), separate from the app schema.
Each run uses `thread_id = run_id`; the checkpointer keys state by thread.

### 2. State serialization — TypedDict refactor (foundation)
The checkpointer serializes graph state, so state must be JSON-serializable.
The current graph carries a mutable `AgentState` object, which does not serialize
cleanly. **Decision: refactor graph state to a LangGraph `TypedDict`** — the
idiomatic model where nodes return partial dicts that LangGraph merges. This makes
all fields (including budget counters, `failed_jobs`, ranking/advice flags, and
the structured judge signals) serialize and survive the checkpoint automatically.
This is the largest and riskiest part; it touches how every node reads/writes
state. It is the make-or-break foundation and must be built and proven first.

### 3. The `human_review` node (the interrupt)
A dedicated node — NOT logic mixed into `do_process_job`, because `interrupt()`
bisects "before pause" and "after resume" and must be an explicit boundary.
`process_job` does the work and flags; if flagged, the router routes to
`human_review`, which calls:

```
value = interrupt({
    "type": "review_request",
    "step_id": step_id,
    "job_title": ...,
    "score": ..., "score_decision": ...,
    "llm_decision": ...,
    "review_reason": ...,
})
decision = value["decision"]   # "Apply" | "Maybe" | "Skip"
comment  = value.get("comment")
# apply, record as authoritative human decision, continue
```

Everything after `interrupt()` runs only after resume.

### 4. Pause/resume coordination across HTTP
The run executes in a FastAPI background task. On interrupt:
- `invoke()` returns (does not block); state is checkpointed; the background task
  ends; the wrapper sets run status `waiting_for_human`.
- The dashboard polls, sees `waiting_for_human`, and shows the review inline using
  the interrupt payload.
- The human submits a decision to a NEW endpoint `POST /runs/{id}/resume`.
- That endpoint calls `graph.invoke(Command(resume={...}), config={thread_id})`,
  which loads the checkpoint and resumes from `human_review`. Processing continues
  (in the resume request or its own background task) and may interrupt again for
  the next flagged job.

## Routing changes

```
route_after_process_job:
    if cancelled            → END (cancelled)
    if this job needs_review → human_review
    if more jobs            → process_job
    else                    → rank_jobs

route_after_human_review (post-resume):
    if more jobs            → process_job
    else                    → rank_jobs
```

## Interaction checklist (what this feature touches)

- **Budget:** a paused run holds no resources. On resume, `llm_calls_made` and
  `max_llm_calls` must be restored from the checkpoint — handled automatically by
  the TypedDict state (component 2).
- **Cancellation:** a `waiting_for_human` run must be cancellable. Either resume
  with a cancel command or mark cancelled and skip resumption.
- **Structured signals & failure accounting:** `judge_status`, `judge_skip_reason`,
  `cache_hit`, `failed_jobs`, ranking/advice flags must all survive the checkpoint
  (another reason for the TypedDict state).
- **Existing post-hoc review:** decide whether to keep it as a fallback alongside
  inline review, or migrate fully to inline.
- **Observability (AgentOps):** record the human decision, the comment, the
  reviewer identity, and the timestamp — and preserve the agent's original
  score/LLM decisions alongside, so the trace shows both what the agent decided
  and how the human overrode it. This is the audit trail the project's thesis
  requires.

## Build sequence (each step testable in isolation)

1. **Checkpointer + TypedDict state refactor.** Get the EXISTING graph working
   with a Postgres checkpointer and NO interrupts yet. Prove a normal run
   completes through the checkpointer and state persists. Riskiest; isolate it.
2. **Add `human_review` node + interrupt.** Make a flagged job pause. Verify
   `invoke()` returns with an interrupt and state is checkpointed.
3. **Resume endpoint + dashboard "awaiting review" UI.** Wire the human decision
   back via `Command(resume=…)`.
4. **Edge cases.** Multi-flag runs, cancellation-while-paused, status transitions,
   and recording the full human-decision audit trail.

## Scope note

This is a multi-session feature. Step 1 alone (checkpointer + state refactor) is
substantial because it rebuilds the state layer. It is the most architecturally
ambitious addition in the project — stateful, cross-cutting, spanning LangGraph,
PostgreSQL, FastAPI, the dashboard, and the observability layer. Gate it behind a
stable core graph (already met) and build it in the sequence above.