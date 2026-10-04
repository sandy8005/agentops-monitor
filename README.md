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
> keep correct. Migration `0013` finishes the move: `runs.mode` defaults to
> `agent` and the database rejects any new non-agent run; every legacy goal record —
> a `pipeline` run, or a goal naming the removed `pool` practice source — is
> **explicitly retired** (`goal_retired_at`, `goal_retired_reason`) and, if it had
> not finished, closed as `failed / engine_retired` (or `cancelled` if the user
> asked). A retired run is never executed or resumed.
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
2. computes the request's **maximum possible cost**: an input-token bound (below),
   plus the `max_output_tokens` cap that every request carries (thinking tokens
   count against it), priced at the **highest rate in effect in the next 15
   minutes** (a request dispatched just before a price change is still bounded);
3. refuses the request unless
   `known spend + bounds of unknown-cost calls + open reservations + this request ≤ cap`;
4. records the reservation.

The reservation is **settled in the same transaction that records the call**, so
spend is never counted twice or not at all. A worker that dies mid-request leaves
its reservation `abandoned`: still counted against the cap and reported as unknown
cost (the request may have been billed). A refused request is a budget stop: the
caller takes its rules fallback, so a capped run keeps working without the model.

**Input-token bound — fail closed.** The byte bound (prompt UTF-8 bytes ÷
`LLM_RESERVE_BYTES_PER_TOKEN`) is a *proven* upper bound because a token never
covers less than one byte; values above `1.0` would under-reserve, so settings
reject them (and `NaN`/`inf`). With `LLM_INPUT_TOKEN_BOUND=provider` (default) the
provider's own `count_tokens` tightens it to `count × (1 + LLM_TOKEN_COUNT_MARGIN) + 16`,
never above the byte bound; any counting error, timeout, missing field or
implausible value falls back to the byte bound. `LLM_INPUT_TOKEN_BOUND=bytes` skips
the extra (free) call.

**Ledger constraints** are in the schema, not only in code: `priced` ⇒ cost set,
`unknown` ⇒ cost `NULL`, `not_billed` ⇒ no charge; no negative cost, token, latency
or bound; `reservation_id` references `llm_cost_reservations` and is unique (a
reservation settles exactly one call); a reservation's `status` and `settled_at`
agree; costs are stored as `NUMERIC(18,10)` so small calls are not rounded to $0.
`SELECT * FROM cost_ledger_violations(<run_id>)` reconciles a run (empty = clean).

**Price overrides are both-or-neither.** `LLM_INPUT_PRICE_PER_MILLION` and
`LLM_OUTPUT_PRICE_PER_MILLION` must be set together; one alone is a startup error
(and treated as an unknown price if it ever reaches the pricing code).

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

### Error-handling policy

A per-item `except Exception` (one job, one tool action, one advice item, the
ranking write, the resume security signal) may record a failure and move on only
for an ordinary item failure: bad data in one posting, an unusable model answer, a
transient provider hiccup. Two classes are never absorbed — `router.must_propagate`
re-raises them so the run fails (or retries) with a classified error code:

* infrastructure failures: database errors and lost execution ownership
  (`database_unavailable` is retryable);
* programming errors in our own code: `TypeError`, `AttributeError`, `NameError`,
  `ImportError`, `AssertionError`, `NotImplementedError`, `RecursionError`.

Model degradation (not configured, quota, outage, invalid output) is classified
separately by `llm.is_degraded_model_error` and takes the rules fallback. API
validation returns 422 only for `ValueError`/pydantic errors; anything else is a
500. The remaining broad handlers are provider-SDK boundaries, best-effort logging
and worker supervision. `tests/test_exception_policy.py` covers each case.

---

## Job sources

Only live providers are in scope:

- **Adzuna** — real role + location search.
- **Remotive** — remote-only feed (location is informational; postings that name a
  candidate region are geo-checked).

Every real provider request is recorded **before dispatch** in
`external_search_attempts` (fenced by execution generation) and closed as
`succeeded`/`failed` afterwards; a request whose worker died is marked `abandoned`
(outcome unknown) by the next generation. A title/provider combination gets at
most 3 real provider requests per run, counted from those rows, so a crash loop
between dispatch and recording cannot call a provider without bound.

Each search is **run-scoped**: a run only ever sees postings its own searches
returned. Rows from any other source (old seed/CSV/scraped data in a legacy database)
are never returned. Duplicates are collapsed URL-first, then by a
title + company + location fingerprint.

**Geographic eligibility** of remote postings is resolved **structurally**
(`geo.py`): text becomes ISO countries + regions from a hierarchy, by longest-phrase
matching ("latin america" never also yields "america"; "south africa" never
"africa"; "Austin, TX" is the US).

```
world ─┬─ americas ─┬─ north_america  (us, ca, mx)
       │            └─ latam          (mx, br, ar, …)
       ├─ emea ─────┬─ europe ── eu   (member states)
       │            ├─ middle_east
       │            └─ africa
       └─ apac
```

A candidate country is eligible when the posting names it or a region containing
it — so a "Germany only" posting rejects a French candidate (the old keyword lists
treated both as "europe"). A candidate who names only a region is eligible if the
posting covers that region or an ancestor, `unknown` if the posting covers part of
it, otherwise ineligible. Ambiguous names ("Georgia"), unresolvable text (a bare
city) and exclusions ("worldwide except US") are `unknown`, never guessed.

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
- **Erasure is monotonic.** Every erased run (resume erasure, run deletion,
  retention) gets an append-only **tombstone** and a new execution generation.
  Database triggers then blank or drop any *late* write for it — trace prompts and
  responses, tool payloads, step context, review payloads, advice, resume
  suggestions, LangGraph checkpoints — so a stale worker that was still finishing
  cannot re-persist erased data; metrics (tokens, cost, status) are kept. An erased
  resume cannot be restored or rewritten and its parse cannot be re-cached (until
  the same text is uploaded again).
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
python migrate.py                            # empty DB -> latest schema (the ONLY schema path;
                                             # `python db_pg.py` is the same entrypoint)
uvicorn api:app                              # process 1: API + dashboard
python worker.py                             # process 2: executes runs
```

`migrate.py` holds a PostgreSQL advisory lock for the whole session (discover → read
applied → apply → record), and the LangGraph checkpoint setup takes the same lock,
so replicas deploying at the same time cannot race a migration. The worker **exits
with status 2** if the checkpoint schema cannot be set up — let your supervisor
(systemd, Docker, Kubernetes) restart it or mark it unhealthy.

### Tests and CI

The suite needs PostgreSQL 16 (`DB_*` settings, migrated with `python migrate.py`).
Model calls are stubbed and tests set their own model settings, so results do not
depend on your `.env`. The markers split it into three disjoint slices:

```bash
pytest -m "not db"            # pure logic, no database
pytest -m "db and not e2e"    # PostgreSQL integration, authorization, pause/resume,
                              # concurrency, cost ledger, erasure guards
pytest -m e2e                 # start -> two human-review pauses -> completion through
                              # the real worker, API, store and Postgres checkpointer
```

Migration paths are checked with the real CLI against throwaway databases (the
`DB_USER` needs `CREATEDB`):

```bash
python scripts/check_migrations.py   # empty DB -> latest; latest re-run is a no-op;
                                     # EVERY older version -> latest, and each upgraded
                                     # schema must be identical to a fresh install
```

`.github/workflows/ci.yml` runs on every push and pull request, top to bottom, and
any failing step fails the run (no `continue-on-error`, no `|| true`):

1. build the archive with `git archive HEAD` and run `scripts/check_release.py` on
   that exact file (it is kept as a build artifact);
2. `python -m compileall`;
3. `pytest -m "not db"` with no database configured;
4. `python migrate.py` on the empty PostgreSQL 16 service database, then again (must
   print "No pending migrations"), then `python db_pg.py --status` (nothing pending);
5. `scripts/check_migrations.py` (every upgrade path);
6. `pytest -m "db and not e2e"` and `pytest -m e2e`, each writing a JUnit report;
7. `scripts/ci_summary.py`, which fails unless the reports together cover **every**
   collected test exactly once with **zero** failures and **zero** skips.

`scripts/check_release.py` also validates the workflow itself: an empty or invalid
`ci.yml`, or one missing any of these stages, fails the release check.

### Packaging

Ship tracked source only — never a working directory (no Explorer / VS Code /
`Compress-Archive` zips: they pick up `.env`, `__pycache__/` and other local files):

```bash
git status                           # commit first: git archive ships HEAD
python scripts/make_release.py       # git archive HEAD -> dist/agentops-monitor-<sha>.zip,
                                     # runs check_release.py on it, prints its SHA-256
```

`make_release.py` refuses a dirty working tree and deletes the archive if the check
fails. The check rejects `.env` / `.env.*` (except `.env.example`), bytecode and
caches, virtualenvs, PDFs/DOCX, missing **or empty** required files, an incomplete CI
workflow, and any value for a secret-looking variable in `.env.example`. To check an
archive built some other way: `python scripts/check_release.py some.zip`.

### Repository layout

- Top level: the runtime (API, worker, agent loop and tools, router, scoring, sources, persistence).
- `migrations/`: schema history; `migrate.py` applies it.
- `static/`: dashboard.
- `tests/`: pytest suite.
- `scripts/eval/`: offline measurement (`fallback_agreement.py`).
- `scripts/make_release.py`: the only supported way to build a release archive.
- `scripts/check_release.py`: release-hygiene check (used by CI and make_release.py).
- `scripts/check_migrations.py`: every migration path ends at the fresh-install schema.
- `scripts/ci_summary.py`: CI proof that every test ran once, green, unskipped.
- `scripts/legacy_migrations/`: pre-runner migration scripts, superseded by `migrations/`.
- `docs/design-history.md`: the original (superseded) human-in-the-loop design.