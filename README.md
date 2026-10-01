# AgentOps Monitor

An AI job-search agent with a built-in observability, evaluation, and human-review layer — plus a web dashboard to watch and control it.

- **The Job Search Agent** — reads a resume, searches live job providers (Adzuna, Remotive), evaluates each posting against the resume, ranks the matches and writes evidence-checked resume suggestions.
- **AgentOps Monitor** — records everything the agent does (every decision, tool call, LLM call, cost, latency, error and retrieved context), evaluates the quality of its decisions, and pauses for human approval where the signals disagree.

The agent is the worker; the Monitor is the observer.

---

## Architecture

- **One execution engine: a bounded controller agent** (`agent_loop.py`), a LangGraph graph `setup → decide → act → review* → finalize` with a **Postgres checkpointer**, so a run can pause for a human and resume later, across processes.
- The API never runs agent work. `POST /runs` creates the run and enqueues a job in the **durable Postgres queue** (`job_queue`) in one transaction. A separate **worker** (`worker.py`) claims jobs with `SELECT … FOR UPDATE SKIP LOCKED` and executes them.
- **Two ownership guarantees.** The queue *lease token* decides who owns the queue row; a PostgreSQL *advisory lock per run* (`run_lock.py`) decides who may execute the run. Every execution takes a new **execution generation**; every write of agent output (actions, searches, rankings, advice, final status) **and every LLM reservation** is fenced by it, so a superseded worker gets `ExecutionLost` instead of overwriting newer results or spending money.
- You run **two processes**: the API (`uvicorn api:app`) and the worker (`python worker.py`).

> The earlier fixed-sequence "pipeline" engine (`autonomous_graph.py`) was **retired**:
> two engines meant two sets of lifecycle, budget, pause/resume and error semantics to
> keep correct. A run still queued for it is closed as `failed / engine_retired`
> without executing; a paused one cannot be resumed (409) and should be cancelled.
> The original design spec is kept, clearly marked historical, in `docs/design-history.md`.

### The controller loop

```
Goal → controller decision → backend validation → ONE tool → compact observation
     → controller decision → … → finalize (outcome verified from persisted results)
```

| Module | Role |
|---|---|
| `agent_goal.py` | The goal, the user's **fixed constraints** (location, work mode, employment type, seniority — the controller can never change them), the hard limits and the run's **search space** (candidate titles). |
| `agent_controller.py` | Proposes the next action: Gemini (structured JSON) or a deterministic rules policy. Backend state (counts, ids, limits, search space) is stated as fact; everything a user, provider or model wrote is fenced as untrusted data. Only actions that are possible **now** are offered. |
| `agent_tools.py` | `search_jobs`, `evaluate_jobs`, `rank_jobs`, `generate_advice`, `request_human_input`, `finish`. Strict argument models, semantic checks against backend state, replay-safe execution. |
| `agent_loop.py` | The graph, the budget, and the guard checked before every decision and every tool call. |
| `agent_store.py` | Persistence and cost accounting; every write presents the execution generation. |
| `router.py` | Shared tool library: resume parsing, per-job evaluation (requirements → score → judge → evaluator → review flags), human decisions, caches. |

**Completion is a backend invariant, not a prompt.** `finish` is accepted only when
the goal is met (verified from persisted results) **or** the search space is
exhausted — the search limit is reached, or every *candidate title × enabled
provider* combination was searched (providers that failed with bad credentials or
missing keys count as unusable) — **and** no discovered eligible job is left
unevaluated. Until then `finish` is neither offered to the model nor accepted from
it. Cancellation, hard limits and repeated tool failures stop a run through the
guard, never through `finish`. A run that never searched can therefore never end
as `no_matches`.

**Outcome classification** (`runs.status` + `runs.error_code`): `success`,
`partial_success`, `no_matches` (`no_matches` when the search space was exhausted,
`limit_reached` when a limit stopped it first), `completed_with_errors`, `cancelled`,
`failed`. When every search failed, the providers' own classification is kept:
`job_source_auth_failed` / `job_source_invalid_response` are terminal,
`job_source_rate_limited` / `job_source_unavailable` are retried. Database errors are
never reported as provider errors: they fail the run as `database_unavailable`
(retried).

### Hard limits

All enforced by the backend: iterations, searches, LLM requests, **active runtime**
and a **hard USD cap**.

**USD cap — reserve, then spend.** Before every provider request (retries
included), one transaction under the run-row lock:

1. proves this worker still owns the run (execution generation);
2. computes the request's **maximum possible cost** — prompt UTF-8 bytes as an
   upper bound on input tokens, plus the `max_output_tokens` cap that every request
   carries (thinking tokens count against it), priced at today's rate;
3. refuses the request unless
   `known spend + bounds of unknown-cost calls + open reservations + this request ≤ cap`;
4. records the reservation.

The reservation is **settled in the same transaction that records the call**, so
spend is never counted twice or not at all. A worker that dies mid-request leaves
its reservation `abandoned`: still counted against the cap and reported as unknown
cost (the request may have been billed). A refused request is a budget stop: the
caller takes its rules fallback, so a capped run keeps working without the model.

**Unknown is never $0.** Every `llm_calls` row has an explicit `cost_status`:

| `cost_status` | when | tokens / `cost_usd` |
|---|---|---|
| `priced` | provider reported usage | from usage; output includes thinking tokens |
| `unknown` | usage metadata missing, or the request failed **after dispatch** (timeout, reset, 5xx) | `NULL`; `cost_upper_bound_usd` = what was reserved |
| `not_billed` | the provider rejected it before doing work (429, 401/403, 400) | `NULL` |

Every endpoint returns `known_cost_usd`, `unknown_cost_calls`, `cost_complete` and
`cost_upper_bound_usd`; `total_cost` is a number only when it is complete. A capped
run with unknown cost *and no bound* (legacy rows) stops with `cost_unknown`.

**Pricing is effective-dated** (`pricing.py`): each model has price windows
(`valid_from`, `valid_until`), and a call is priced with the window containing the
moment it was made. `gemini-3.6-flash`: $0.75 / $3.75 per 1M input / output tokens
through 2026-12-31, $1.50 / $7.50 from 2027-01-01 (Google's published rates, checked
2026-09-29). Add a new window when prices change — never edit an old one. A cost cap
with a model that has no price for today is refused (`422`) or stopped
(`cost_unknown`).

**Active runtime.** Only execution time is charged: waiting for a human or in the
retry queue is not. If a worker dies mid-execution, its open interval is charged up
to its **last heartbeat plus `EXECUTION_HEARTBEAT_GRACE_SECONDS`** (default 45s; the
heartbeat runs every 30s) — a conservative bound, since the exact moment it stopped
is unknowable.

### Audit model

```
Run
└── execution generation (one per worker execution)
    └── controller iteration
        ├── decision            agent_actions          (replayed, never re-asked)
        │   └── attempts        agent_action_attempts  (one per generation that executed it)
        ├── search              agent_searches         (one row per generation)
        └── LLM requests        llm_calls + llm_cost_reservations
```

A search that failed **transiently** (429, 5xx, network) is genuinely refetched by
the next generation; a successful search is reused; configuration failures are not
retried.

### Per-job evaluation

For each discovered job: **requirements** (LLM extraction, cached durably with
provenance; a rules fallback is used only when the *model* is degraded — database
and programming errors fail visibly) → deterministic **match score** → **judge**
(LLM Apply/Maybe/Skip, only in the uncertain 20–80 band) → optional **evaluator** →
if flagged for any reason, the run **pauses for human review**. `rank_jobs` sorts by
decision bucket (Apply > Maybe > Skip), then score. `generate_advice` writes
evidence-checked resume suggestions (below).

### Human approval workflow

A flagged job creates an immutable `review_requests` row and the graph pauses
(`interrupt()`); the run becomes `waiting_for_human`. The reviewer answers in the
dashboard (`POST /runs/{id}/resume` with the `review_id` of the card they saw — a
stale tab gets 409). The human's decision becomes the authoritative
`final_decision`, recorded with who decided and when.

A resume job always leaves the run in a **deterministic** state: waiting (if the
checkpoint is paused on a different review), repaired from a completed checkpoint,
continued (if the review was already applied and a previous worker died
mid-graph), or `failed / checkpoint_error` — never left `running`.

### Resume suggestions

```
resume evidence → rule suggestions → optional Gemini wording → rewrite validator
               → structured persistence (resume_suggestions) → API → UI
```

Only verified resume passages are sent to the model. `validate_rewrite()` rejects a
rewrite that adds technologies not evidenced in the resume, numbers, dates, names or
employers, credentials (certifications, degrees, licences), flips a negation, or
turns "learning X" into "experienced in X"; responsibility inflation and any new
wording need the user's confirmation. Rejected suggestions are never returned.

---

## Security

- **Authentication** — session-cookie login, bcrypt hashes. Login for an unknown
  username does the same bcrypt work as a wrong password (no username enumeration by
  timing). Sign-up inserts first and relies on the UNIQUE constraint (no
  check-then-insert race).
- **Authorization** — every resume and run is owner-scoped. There is no RBAC.
- **CSRF** — session-bound synchronizer token on every state-changing request.
- **Rate limits** — login, run creation and upload, per IP (`RATE_LIMIT_*`; share
  them across processes with `RATE_LIMIT_STORAGE_URI`).
- **Uploads** — PDF only; read in 64 KB chunks and rejected with 413 as soon as
  `MAX_UPLOAD_BYTES` (default 5 MB) is exceeded; parsed in a resource-limited child
  process. **Also set the body limit at the reverse proxy** (e.g. nginx
  `client_max_body_size 6m`): it is the only layer that can refuse bytes before they
  are received.
- **Prompt injection** — all untrusted text (resume, postings, observations, human
  answers) is fenced in every prompt with a hardening preamble. Detection
  (`prompt_safety.py`) returns stable **pattern IDs with a severity** and never the
  matched text. Policy: a **high**-severity match in a job posting ("ignore previous
  instructions", a forged `"decision": "Apply"`) requests human review of that job;
  **low**-severity phrases that occur in legitimate AI/security job ads ("system
  prompt", "assistant:", "you are now") are only recorded as a security signal; a
  resume is never routed to review. Fencing is a mitigation, not a security boundary.
- **Trace redaction** — `REDACT_TRACE_PAYLOADS` (on by default) hides prompts,
  responses, tool I/O, context and free-text errors in API responses. Display control,
  not deletion.

### Logging policy

Application logs carry **metadata only**: ids, counts, statuses, error codes,
exception *types* and prompt-safety pattern IDs — never resume- or job-derived text
(titles, descriptions, skills, advice, prompts, responses, provider or database error
messages). Application logs are outside the trace-retention purge, so this is
enforced at the call sites (`logging_config.py`).

---

## Job sources

Only live providers are in scope:

- **Adzuna** — real role + location search.
- **Remotive** — remote-only feed (location is informational; postings that name a
  candidate region are geo-checked).

Each search is **run-scoped**: a run only ever sees postings its own searches
returned. Rows from any other source (old seed/CSV/scraped data in a legacy database)
are never returned. Duplicates are collapsed URL-first, then by a
title + company + location fingerprint.

**Geographic eligibility** of remote postings uses a region hierarchy:

```
north_america ── us, canada
emea ─┬─ europe ── uk
      ├─ middle_east
      └─ africa
```

A posting open to a region accepts a candidate in any sub-region ("EMEA" accepts
the UAE; "Europe" accepts the UK). Siblings never match (a Middle East posting does
not accept a German candidate). A candidate who names only a broader region ("EMEA")
against a narrower posting ("Europe") is `unknown`, not `ineligible`.

---

## Scoring is heuristic

The deterministic score (alias normalization, whole-word matching, negation and
"currently learning" handling, any-of groups, experience discrepancy, full
breakdown/provenance) is engineered carefully, but its **category weights and the
Apply ≥ 75 / Maybe ≥ 55 thresholds are hand-selected** and have not been calibrated.
Treat scores as a ranking signal, not an objectively validated match percentage.

Before relying on them, build a labeled benchmark (500–1,000 resume/job pairs with
human labels across role categories, seniority, location/work-authorization cases and
required/preferred skill cases) and measure precision, recall, false-Apply and
false-Skip rates, reviewer agreement and calibration by score band.

The **rules-based requirements fallback** (`rule_requirements.py`) is a degraded
path: it reads "N years" per sentence (skipping preferred, conditional and
non-experience mentions) but cannot build any-of groups. Extractions are tagged
`requirements_method = rule_based`, and `require_verified_matches` excludes matches
that rest only on it. Measure its drift on your data with:

```bash
python scripts/eval/fallback_agreement.py --limit 500 --resume-id <id>
```

---

## Data handling

- **Deleting a resume erases it** (text, name and every trace payload of runs that
  used it, including checkpoints). A resume used by an active run can't be erased (409).
- **Deleting a run** hard-deletes it and its traces.
- **Retention** — the worker purges trace payloads of runs that ended more than
  `TRACE_RETENTION_DAYS` (default 30) days ago.
- **Never commit real resumes.** `.gitignore` excludes `*.pdf`; tests use synthetic
  text. Out of scope for this code: encryption at rest, backup retention, log
  retention, database access control.

---

## Running it

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements-dev.txt          # runtime + test dependencies
cp .env.example .env                         # then fill in DB_*, SESSION_SECRET, GEMINI_API_KEY
python migrate.py                            # empty DB -> latest schema (the ONLY schema path)
uvicorn api:app                              # process 1: API + dashboard
python worker.py                             # process 2: executes runs
```

`migrate.py` holds a PostgreSQL advisory lock for the whole session (discover → read
applied → apply → record), and the LangGraph checkpoint setup takes the same lock,
so replicas deploying at the same time cannot race a migration. The worker **exits
with status 2** if the checkpoint schema cannot be set up — let your supervisor
(systemd, Docker, Kubernetes) restart it or mark it unhealthy.

### Tests and CI

The suite needs PostgreSQL (`DB_*` settings, migrated with `python migrate.py`).
Model calls are stubbed and tests set their own model settings, so results do not
depend on your `.env`. `pytest -m "not db"` runs the pure-logic subset.

`.github/workflows/ci.yml` runs, on every push and pull request: byte-compilation,
the pure-logic tests, migration of an **empty** database, a re-run of the migrations
(must be a no-op), and the full suite (PostgreSQL integration, API authorization,
pause/resume, concurrency and cost-ledger tests) against PostgreSQL 16.

### Packaging

Ship tracked source only — never a working directory:

```bash
git archive --format=zip -o agentops-monitor.zip HEAD
```

### Repository layout

- Top level: the runtime (API, worker, agent loop and tools, router, scoring, sources, persistence).
- `migrations/`: schema history; `migrate.py` applies it.
- `static/`: dashboard.
- `tests/`: pytest suite.
- `scripts/eval/`: offline measurement (`fallback_agreement.py`).
- `scripts/manual/`: print-based manual checks (not collected by pytest; pass them a synthetic resume path).
- `scripts/legacy_migrations/`: pre-runner migration scripts, superseded by `migrations/`.
- `docs/design-history.md`: the original (superseded) human-in-the-loop design.