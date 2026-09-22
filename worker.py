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
import traceback

import job_queue
from autonomous_graph import run_agent_graph, resume_agent_graph
from logging_config import get_logger

log = get_logger("worker")

POLL_INTERVAL = 2.0        # seconds to sleep when the queue is empty
HEARTBEAT_INTERVAL = 30.0  # seconds between heartbeats during a running job
ORPHAN_SWEEP_INTERVAL = 60.0  # seconds between orphan-recovery sweeps


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
    """Run one job under a heartbeat, updating the queue on success/failure.

    All queue mutations are guarded by the per-claim lease token: if orphan recovery
    reclaimed this job and another worker took it, our heartbeat/mark_done/mark_failed
    match zero rows and we stop touching the record — the owning worker is now
    authoritative. (This bounds the queue-record damage; it does not undo agent side
    effects this worker may already have written before losing the lease.)"""
    stop = threading.Event()
    lost_lease = threading.Event()
    lease = job["lease_token"]

    def _beat():
        while not stop.wait(HEARTBEAT_INTERVAL):
            try:
                if not job_queue.heartbeat(job["id"], lease):
                    # We no longer own this job — reclaimed as an orphan and taken by
                    # another worker. Stop heart-beating and flag it; from here on we
                    # must not mutate the queue record.
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
        _run_job(job)
        if lost_lease.is_set() or not job_queue.mark_done(job["id"], lease):
            log.warning("job %s finished but its lease was lost — NOT marking done "
                        "(another worker owns it)", job["id"], extra={"run_id": job["run_id"]})
        else:
            log.info("job %s (%s) done", job["id"], job["kind"], extra={"run_id": job["run_id"]})
    except Exception as e:
        traceback.print_exc()
        if lost_lease.is_set():
            log.error("job %s (%s) errored AFTER losing its lease — leaving it to the "
                      "owning worker: %s", job["id"], job["kind"], e, extra={"run_id": job["run_id"]})
        else:
            outcome = job_queue.mark_failed(
                job["id"], e, job["attempts"], job["max_attempts"], lease)
            state = {"requeued": "requeued for retry",
                     "failed": "FAILED (out of attempts)",
                     "lost": "lease lost — left to owning worker"}.get(outcome, outcome)
            log.error("job %s (%s) errored: %s — %s", job["id"], job["kind"], e, state, extra={"run_id": job["run_id"]})
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