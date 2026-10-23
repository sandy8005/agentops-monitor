# Production readiness checklist

Run these in order for every deploy. Each step names its exact command and what
"pass" means. Stop at the first failure. README.md is the spec; this page is only
the sequence.

## 1. Build from a committed tree

```bash
git status                                   # must be clean: releases ship HEAD
python scripts/make_release.py               # git archive HEAD + check_release.py,
                                             # prints the archive's SHA-256
```

Pass: the script prints the archive path and SHA-256. It refuses a dirty tree and
deletes an archive that fails the check.

## 2. The CI sequence, locally or in CI

The same stages as `.github/workflows/ci.yml`. Use PostgreSQL 16; `DB_USER` needs
`CREATEDB` for step 2.6.

```bash
git archive --format=zip -o /tmp/agentops.zip HEAD
python scripts/check_release.py /tmp/agentops.zip                 # 2.1 release hygiene
python -m compileall -q .                                         # 2.2 byte-compile
DB_NAME= DB_HOST= python -m pytest -m "not db" --junitxml=reports/pure.xml   # 2.3
python migrate.py                                                 # 2.4 empty DB -> latest
python migrate.py | grep "No pending migrations"                  # 2.5 re-run is a no-op
python db_pg.py --status                                          #     nothing PENDING
python scripts/check_migrations.py                                # 2.6 every upgrade path
python -m pytest -m "db and not e2e" --junitxml=reports/db.xml    # 2.7 integration
python -m pytest -m e2e --junitxml=reports/e2e.xml                # 2.8 end to end + recovery
python scripts/ci_summary.py reports/pure.xml reports/db.xml reports/e2e.xml   # 2.9
```

Pass: every command exits 0. Step 2.9 fails on any failure, any skip, or any
collected test missing from the reports.

What 2.7–2.8 prove beyond unit logic:

* **Recovery** (`tests/test_recovery_e2e.py`): a worker killed mid-tool leaves its
  job `running`; after the lease expires, orphan recovery requeues it and a new
  execution generation resumes from the Postgres checkpoint with no duplicate
  decisions, searches, provider requests, evaluations or charges. The superseded
  worker's late writes get `ExecutionLost`. A run waiting for a human survives a
  worker restart and resumes correctly.
* **Gemini contract** (`tests/test_gemini_contract.py`): every prompt type against
  recorded valid, malformed, missing-field, safety-blocked, 429, 5xx and timeout
  responses: the classification, the rules fallback, and one ledger row per HTTP
  attempt with every reservation settled and `cost_ledger_violations` empty.

## 3. Migrate production

```bash
python migrate.py --status                   # review what is PENDING
python migrate.py                            # holds an advisory lock: replicas can't race
python migrate.py --status                   # nothing PENDING
```

Migrate before starting the new API or worker processes. The worker exits with
status 2 if the checkpoint schema can't be set up.

## 4. Live smoke test (real Gemini key, real network)

```bash
python scripts/smoke_live.py                 # one run, max_cost_usd 0.05, <= 6 model calls
python scripts/smoke_live.py --max-cost 0.02 --max-llm-calls 4
```

Pass: exit 0 and `smoke_live: PASSED`. It asserts a terminal non-failed status, at
least one model call, no open reservation, committed spend within the cap and an
empty `cost_ledger_violations(run_id)`. Exit 2 means it didn't start (no real
key, database unreachable or not migrated) and made no request. Exit 1 lists the
failed checks. It prints metadata only. The synthetic `smoke_<hex>` user, resume
and run stay in the database for inspection.

## 5. Process supervision and restart policy

Run the two processes separately, each under a supervisor (systemd, Docker,
Kubernetes):

| Process | Command | Restart policy |
|---|---|---|
| API + dashboard | `uvicorn api:app` | always; it holds no run state |
| Worker | `python worker.py` | always, with backoff (for example systemd `Restart=always`, `RestartSec=5`). Exit status 2 (checkpoint schema setup failed) means the DB isn't ready or migrated: alert on repeated exits instead of restarting forever. |

Why restarting the worker is safe:

* claims use `FOR UPDATE SKIP LOCKED` plus a lease token, and execution takes a
  per-run advisory lock, so two workers never execute one run;
* a killed worker's job is reclaimed after `ORPHAN_AFTER` (15 minutes without a
  heartbeat) by the next worker's startup or periodic `reclaim_orphans()`; the run
  goes `retrying` and resumes from its checkpoint under a new execution generation;
* open cost reservations of the dead generation become `abandoned` and keep
  counting against the USD cap (fail closed);
* runs in `waiting_for_human` aren't touched by recovery; they resume when the
  reviewer answers.

The worker has no graceful-shutdown handler, so any stop, SIGTERM included, acts
like a kill: safe, as above, but the affected run waits up to `ORPHAN_AFTER` before
another worker resumes it. Deploy when few runs are executing, or accept that
latency.

## 6. After the deploy

* `GET /runs` for recent runs: none stuck in `running` past `ORPHAN_AFTER`.
* `SELECT * FROM cost_ledger_violations(<run_id>)` is empty for recent runs.
* Worker logs: metadata only (ids, counts, codes). Alert on repeated
  `llm_unavailable` or `database_unavailable` error codes.
