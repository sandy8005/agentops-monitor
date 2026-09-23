"""
Durable job worker.

Run this ALONGSIDE the API (`python worker.py`). It's the process that actually
executes agent work, pulled from the Postgres-backed job_queue — so work survives
API restarts and crashes (the old FastAPI BackgroundTasks ran in the web process
and were lost on restart).

Loop:
  1. On startup, reclaim orphaned jobs (a previous worker may have died mid-run).
  2. Claim the oldest queued job (FOR UPDATE SKIP LOCKED — race-free across workers).
  3. Run it in a heartbeat-wrapped thread so a long-but-healthy job isn't mistaken
     for an orphan.
  4. On success mark done; on error retry (back to queued) or fail after max attempts.
  5. Sleep briefly when the queue is empty; periodically re-check for orphans.

You can run MULTIPLE workers for parallelism — SKIP LOCKED guarantees they never
claim the same job.
"""
import time
import threading

import job_queue
from autonomous_graph import run_agent_graph, resume_agent_graph
from error_codes import ErrorCode, classify_exception
from database import get_connection
from logging_config import get_logger

log = get_logger("worker")

POLL_INTERVAL = 2.0        # seconds to sleep when the queue is empty
HEARTBEAT_INTERVAL = 30.0  # seconds between heartbeats during a running job
ORPHAN_SWEEP_INTERVAL = 60.0  # seconds between orphan-recovery sweeps

# --- Worker outcome contract -------------------------------------------------
# The queue outcome is decided by the RUN'S recorded status/error_code, NOT by
# "did _run_job raise". Both graph entrypoints catch their own exceptions and
# finalize the run (finish_run / waiting_for_human) before returning normally, so a
# normal return can still mean the run failed — inspecting the run row is the only
# reliable signal.
SUCCESS = "success"
RETRYABLE_FAILURE = "retryable_failure"
TERMINAL_FAILURE = "terminal_failure"

# Failures worth retrying: transient / infrastructure codes. Everything else (bad
# input, parse failure, budget spent, cancelled) is terminal — retrying won't help.
RETRYABLE_ERROR_CODES = {ErrorCode.LLM_UNAVAILABLE, ErrorCode.LLM_QUOTA_EXHAUSTED}


def _mark_run_retrying(run_id):
    """During retry backoff, show the run as 'retrying' (not 'failed') so the monitor
    and users don't see a 'failed' run that's actually about to run again. Keeps the
    last attempt's error_code/stop_reason. No-op if the run is already terminal."""
    with get_connection() as conn:
        conn.cursor().execute(
            "UPDATE runs SET status='retrying' "
            "WHERE id=%s AND status IN ('queued','running','failed')", (run_id,))


def _finalize_run_failed_if_active(run_id, error_code):
    """Finalize a run as 'failed' if it is still in an active (non-terminal) state.
    Covers the case where _run_job raised BEFORE the graph could finalize the run
    (malformed queue payload, dispatch-level error): the queue job is terminally
    failed but the run would otherwise linger in queued/running/retrying. Conditional
    on the current status, so it never overwrites a run the graph already finalized."""
    with get_connection() as conn:
        conn.cursor().execute(
            "UPDATE runs SET status='failed', ended_at=NOW(), error_code=%s, "
            "stop_reason=COALESCE(stop_reason, 'job failed before the run was finalized') "
            "WHERE id=%s AND status IN ('queued','running','retrying')",
            (str(error_code) if error_code else None, run_id))


def _run_outcome(run_id):
    """
    Map a finished job to a queue outcome by reading the RUN'S authoritative status
    and error_code from the DB. This is the worker contract — SUCCESS /
    RETRYABLE_FAILURE / TERMINAL_FAILURE — rather than "did the call raise".

    A paused run (waiting_for_human) means THIS job finished its work; the run
    continues later via a separate resume_run job, so it counts as job success.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT status, error_code FROM runs WHERE id = %s", (run_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return TERMINAL_FAILURE, None   # run vanished — nothing to retry into
    status, error_code = row
    if status in ("success", "no_matches", "completed_with_errors",
                  "cancelled", "waiting_for_human"):
        return SUCCESS, error_code
    if status == "failed":
        if error_code in RETRYABLE_ERROR_CODES:
            return RETRYABLE_FAILURE, error_code
        return TERMINAL_FAILURE, error_code
    # 'running'/'queued'/unknown: the graph didn't finalize (shouldn't happen on a
    # normal return). Treat as retryable so the job isn't silently marked done.
    return RETRYABLE_FAILURE, error_code


def _run_job(job):
    """Dispatch a claimed job to the right agent entrypoint. Runs synchronously in
    the worker (this is the worker's whole purpose)."""
    kind = job["kind"]
    p = job["payload"]
    if kind == "start_run":
        run_agent_graph(
            resume_id=p["resume_id"],
            target_role=p.get("target_role"),
            location=p.get("location"),
            work_mode=p.get("work_mode"),
            employment_type=p.get("employment_type"),
            evaluate=p.get("evaluate", False),
            run_id=p["run_id"],
            live_only=p.get("live_only", False),
        )
    elif kind == "resume_run":
        resume_agent_graph(p["run_id"], p["decision"], p.get("comment", ""))
    else:
        raise ValueError(f"unknown job kind: {kind}")


def _process(job):
    """Run one job under a heartbeat, then record its outcome on the queue.

    Two-part contract:
      1. Outcome is SUCCESS / RETRYABLE_FAILURE / TERMINAL_FAILURE, decided by the
         run's recorded status (see _run_outcome) — not by whether _run_job raised,
         since the graph catches its own errors and finalizes the run before
         returning. A raised exception is itself a failure, classified by its code.
      2. Every queue mutation is lease-guarded: if orphan recovery reclaimed this
         job and another worker took it, our heartbeat/mark_* match zero rows and we
         stop touching the record. (Bounds queue-record damage; does not undo agent
         side effects already written before the lease was lost.)
    """
    stop = threading.Event()
    lost_lease = threading.Event()
    lease = job["lease_token"]

    def _beat():
        while not stop.wait(HEARTBEAT_INTERVAL):
            try:
                if not job_queue.heartbeat(job["id"], lease):
                    lost_lease.set()
                    log.warning("job %s lease lost — another worker owns it now; "
                                "this worker will stop touching the queue record",
                                job["id"], extra={"run_id": job["run_id"]})
                    return
            except Exception:
                pass  # heartbeat failure shouldn't kill the job

    beat = threading.Thread(target=_beat, daemon=True)
    beat.start()
    try:
        # Decide the outcome. A raised exception is classified by its error code; a
        # normal return is classified from the run's finalized status.
        try:
            _run_job(job)
            if job.get("run_id") is not None:
                outcome, code = _run_outcome(job["run_id"])
            else:
                outcome, code = SUCCESS, None
        except Exception as e:
            code = classify_exception(e)
            outcome = RETRYABLE_FAILURE if code in RETRYABLE_ERROR_CODES else TERMINAL_FAILURE
            log.exception("job %s (%s) raised (code=%s)", job["id"], job["kind"], code,
                          extra={"run_id": job["run_id"]})

        # Record the outcome — only if we still hold the lease.
        if lost_lease.is_set():
            log.warning("job %s: lease lost — NOT recording outcome '%s' (another worker owns it)",
                        job["id"], outcome, extra={"run_id": job["run_id"]})
        elif outcome == SUCCESS:
            if job_queue.mark_done(job["id"], lease):
                log.info("job %s (%s) done", job["id"], job["kind"], extra={"run_id": job["run_id"]})
            else:
                log.warning("job %s finished but lease was lost — NOT marking done",
                            job["id"], extra={"run_id": job["run_id"]})
        else:
            terminal = (outcome == TERMINAL_FAILURE)
            res = job_queue.mark_failed(job["id"], code or outcome, job["attempts"],
                                        job["max_attempts"], lease, terminal=terminal)
            log.error("job %s (%s) failed [%s, code=%s] — %s",
                      job["id"], job["kind"], "terminal" if terminal else "retryable",
                      code, res, extra={"run_id": job["run_id"]})
            # Keep the RUN's status consistent with the QUEUE outcome.
            if job.get("run_id") is not None:
                if res == "requeued":
                    # Will retry after backoff — show 'retrying', not the graph's
                    # 'failed', while keeping the last attempt's error_code.
                    _mark_run_retrying(job["run_id"])
                elif res == "failed":
                    # Terminal/exhausted — make sure the run is finalized too (it may
                    # still be queued/running if _run_job raised before the graph
                    # could finalize). No-op if already terminal.
                    _finalize_run_failed_if_active(job["run_id"], code)
                # res == "lost": another worker owns it now — don't touch the run.
    finally:
        stop.set()


def main():
    log.info("worker %s starting", job_queue.WORKER_ID)
    # Startup orphan recovery: a previous worker may have died mid-job.
    try:
        n = job_queue.reclaim_orphans()
        if n:
            log.info("reclaimed %s orphaned job(s) on startup", n)
    except Exception as e:
        log.warning("orphan recovery failed on startup: %s", e)

    last_sweep = time.time()
    while True:
        try:
            # Periodic orphan sweep (in case a sibling worker died).
            if time.time() - last_sweep > ORPHAN_SWEEP_INTERVAL:
                try:
                    job_queue.reclaim_orphans()
                except Exception:
                    pass
                last_sweep = time.time()

            job = job_queue.claim_next()
            if job is None:
                time.sleep(POLL_INTERVAL)
                continue
            log.info("claimed job %s: %s (attempt %s/%s)", job["id"], job["kind"], job["attempts"], job["max_attempts"], extra={"run_id": job["run_id"]})
            _process(job)
        except KeyboardInterrupt:
            log.info("worker stopping (Ctrl+C)")
            break
        except Exception as e:
            # A failure in the loop itself (e.g. DB blip) — log and keep going.
            log.error("worker loop error: %s", e)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()