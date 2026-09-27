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
import run_lock

log = get_logger("worker")

POLL_INTERVAL = 2.0        # seconds to sleep when the queue is empty
HEARTBEAT_INTERVAL = 30.0  # seconds between heartbeats during a running job
ORPHAN_SWEEP_INTERVAL = 60.0  # seconds between orphan-recovery sweeps
RETENTION_SWEEP_INTERVAL = 3600.0  # seconds between trace-retention purges

# --- Worker outcome contract -------------------------------------------------
# The queue outcome is decided by the RUN'S recorded status/error_code, NOT by
# "did _run_job raise". Both graph entrypoints catch their own exceptions and
# finalize the run (finish_run / waiting_for_human) before returning normally, so a
# normal return can still mean the run failed — inspecting the run row is the only
# reliable signal.
SUCCESS = "success"
RETRYABLE_FAILURE = "retryable_failure"
TERMINAL_FAILURE = "terminal_failure"

# Failures worth retrying: TRANSIENT infrastructure codes only. A per-minute rate
# limit or an unavailable provider clears on its own; an exhausted daily/project
# QUOTA does not recover on a 10-40s backoff, so it is terminal (retrying would only
# burn attempts). Everything else (bad input, parse failure, budget, cancel) is
# terminal too.
RETRYABLE_ERROR_CODES = {ErrorCode.LLM_UNAVAILABLE, ErrorCode.LLM_RATE_LIMITED,
                         # An Adzuna/Remotive rate limit or outage is recoverable too;
                         # bad credentials / malformed responses are not.
                         ErrorCode.JOB_SOURCE_RATE_LIMITED, ErrorCode.JOB_SOURCE_UNAVAILABLE}
# Log a repeating background failure (heartbeat, orphan sweep) at most this often,
# so a DB outage leaves a diagnostic trail without flooding the log.
WARN_EVERY_SECONDS = 60.0


class _RateLimitedWarner:
    """Emit the first warning immediately, then at most one per interval, reporting
    how many failures were suppressed in between."""

    def __init__(self, interval=WARN_EVERY_SECONDS):
        self.interval = interval
        self._last = 0.0
        self._suppressed = 0
        self._lock = threading.Lock()

    def warn(self, msg, *args, **kwargs):
        with self._lock:
            now = time.time()
            if now - self._last < self.interval:
                self._suppressed += 1
                return
            suppressed, self._suppressed, self._last = self._suppressed, 0, now
        if suppressed:
            msg = msg + " (%d similar failure(s) suppressed)"
            args = args + (suppressed,)
        log.warning(msg, *args, **kwargs)

# How long to wait before re-offering a job whose run is still locked by another
# worker (not counted as an attempt).
LOCK_BUSY_DELAY = 15


def _mark_run_retrying(run_id):
    """During retry backoff, show the run as 'retrying' (not 'failed') so the monitor
    and users don't see a 'failed' run that's actually about to run again. Keeps the
    last attempt's error_code/stop_reason for context.

    ended_at is CLEARED (a retrying run has not ended) and the failed attempt's end
    time moves to last_attempt_ended_at — otherwise the row would read
    "status=retrying, ended_at=10:15", which contradicts itself."""
    with get_connection() as conn:
        conn.cursor().execute(
            "UPDATE runs SET status='retrying', "
            "  last_attempt_ended_at = COALESCE(ended_at, NOW()), ended_at = NULL "
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
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status, error_code FROM runs WHERE id = %s", (run_id,))
        row = cur.fetchone()
    if not row:
        return TERMINAL_FAILURE, None   # run vanished — nothing to retry into
    status, error_code = row
    if status in ("success", "partial_success", "no_matches", "completed_with_errors",
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
    if _run_mode(job.get("run_id") or p.get("run_id")) == "agent":
        import agent_loop
        if kind == "start_run":
            agent_loop.run_agent_loop(p["run_id"], queue_attempt=job.get("attempts", 1))
        elif kind == "resume_run":
            agent_loop.resume_agent_loop(p["run_id"], p, queue_attempt=job.get("attempts", 1))
        else:
            raise ValueError(f"unknown job kind: {kind}")
        return
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
        resume_agent_graph(p["run_id"], p["decision"], p.get("comment", ""),
                           reviewer_user_id=p.get("reviewer_user_id"),
                           reviewer=p.get("reviewer"),
                           queue_attempt=job.get("attempts", 1),
                           review_id=p.get("review_id"))
    else:
        raise ValueError(f"unknown job kind: {kind}")


def _run_mode(run_id):
    """'agent' or 'pipeline' (pre-0010 databases have no mode column -> pipeline)."""
    if run_id is None:
        return "pipeline"
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT mode FROM runs WHERE id = %s", (run_id,))
            row = cur.fetchone()
        return (row[0] if row else None) or "pipeline"
    except Exception:
        return "pipeline"


def _job_is_stale(job):
    """
    True if this job no longer has anything to do, because the run already moved on
    — typically a job that orphan recovery requeued while its original worker was
    merely slow, and that worker then finished the run. Executing it again would
    duplicate the whole run (start_run) or apply a stale decision to the NEXT review
    (resume_run). Checked AFTER taking the run lock, so the answer can't change
    underneath us.
    """
    run_id = job.get("run_id")
    if run_id is None:
        return False
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status, error_code FROM runs WHERE id = %s", (run_id,))
        row = cur.fetchone()
    if not row:
        return True
    status, error_code = row
    if status in ("queued", "retrying", "running"):
        return False
    # R02: a run left 'failed' with a RETRYABLE code by an older (non-atomic) crash
    # still has a legitimate retry pending — do not close it without executing.
    if status == "failed" and error_code in RETRYABLE_ERROR_CODES:
        return False
    # Anything else (finished, cancelled, paused for a human) means the run moved on.
    # (resume_run decisions are additionally bound to their review_id in the graph.)
    return True


def _process(job):
    """Execute one claimed job under the RUN-LEVEL execution lock (see run_lock).

    The queue lease decides who owns the queue ROW; the run lock decides who may
    EXECUTE the run. If another worker still holds the run (e.g. the job was
    reclaimed as an orphan while its original worker was merely slow), this worker
    hands the job back without consuming an attempt instead of running it in
    parallel."""
    run_id = job.get("run_id")
    if run_id is None:
        return _process_locked(job, None)
    lock = run_lock.RunLock.try_acquire(run_id)
    if lock is None:
        released = job_queue.release(job["id"], job["lease_token"], LOCK_BUSY_DELAY)
        log.warning("job %s: run %s is being executed by another worker — %s",
                    job["id"], run_id,
                    "handed back to the queue" if released else "lease already lost",
                    extra={"run_id": run_id})
        return
    run_lock.clear(run_id)
    try:
        if _job_is_stale(job):
            if job_queue.mark_done(job["id"], job["lease_token"]):
                log.info("job %s: run %s already moved on — stale job closed without executing",
                         job["id"], run_id, extra={"run_id": run_id})
            return
        _process_locked(job, lock)
    finally:
        lock.release()
        run_lock.clear(run_id)


def _process_locked(job, lock):
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
    hb_warner = _RateLimitedWarner()

    def _lose(reason):
        lost_lease.set()
        if job.get("run_id") is not None:
            # Cooperative stop: the agent aborts at its next node boundary and the
            # graph entrypoints skip finalization (another worker owns the run now).
            run_lock.mark_lost(job["run_id"])
        log.warning("job %s: %s — stopping execution", job["id"], reason,
                    extra={"run_id": job["run_id"]})

    def _beat():
        while not stop.wait(HEARTBEAT_INTERVAL):
            if lock is not None and not lock.alive():
                _lose("run-lock connection lost (lock released by the server)")
                return
            try:
                if not job_queue.heartbeat(job["id"], lease):
                    _lose("queue lease lost")
                    log.warning("job %s lease lost — another worker owns it now; "
                                "this worker will stop touching the queue record",
                                job["id"], extra={"run_id": job["run_id"]})
                    return
            except Exception as e:
                # A transient heartbeat failure must not kill the job — but it MUST be
                # visible: a DB problem that keeps heartbeats failing makes the job
                # look stale and triggers orphan recovery, and this is the trail
                # that explains why.
                hb_warner.warn("job %s: heartbeat failed (%s: %s) — will retry; "
                               "job may be reclaimed as an orphan if this persists",
                               job["id"], type(e).__name__, e,
                               extra={"run_id": job["run_id"]})

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
            # R02: queue row + run status change in ONE lease-guarded transaction.
            res = job_queue.fail_job_and_run(job["id"], lease, job.get("run_id"),
                                             code or outcome, job["attempts"],
                                             job["max_attempts"], terminal=terminal,
                                             error_code=code)
            log.error("job %s (%s) failed [%s, code=%s] — %s",
                      job["id"], job["kind"], "terminal" if terminal else "retryable",
                      code, res, extra={"run_id": job["run_id"]})
    finally:
        stop.set()


def main():
    log.info("worker %s starting", job_queue.WORKER_ID)
    # LangGraph checkpoint schema: normally created by `python migrate.py`; run once
    # here as a safety net — NOT on every graph execution (it's DDL).
    try:
        from checkpointing import setup_schema
        setup_schema()
    except Exception as e:
        log.error("langgraph checkpoint schema setup failed: %s", e)
    sweep_warner = _RateLimitedWarner()
    # Startup orphan recovery: a previous worker may have died mid-job.
    try:
        n = job_queue.reclaim_orphans()
        if n:
            log.info("reclaimed %s orphaned job(s) on startup", n)
    except Exception as e:
        log.warning("orphan recovery failed on startup: %s", e)

    last_sweep = time.time()
    last_retention = 0.0
    while True:
        try:
            # Periodic trace-retention purge (settings.trace_retention_days).
            if time.time() - last_retention > RETENTION_SWEEP_INTERVAL:
                try:
                    from privacy import purge_expired_traces
                    from settings import settings
                    purge_expired_traces(settings.trace_retention_days)
                except Exception as e:
                    log.warning("trace retention purge failed: %s", e)
                last_retention = time.time()

            # Periodic orphan sweep (in case a sibling worker died).
            if time.time() - last_sweep > ORPHAN_SWEEP_INTERVAL:
                try:
                    n = job_queue.reclaim_orphans()
                    if n:
                        log.info("orphan sweep reclaimed %s job(s)", n)
                except Exception as e:
                    sweep_warner.warn("periodic orphan sweep failed (%s: %s)",
                                      type(e).__name__, e)
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