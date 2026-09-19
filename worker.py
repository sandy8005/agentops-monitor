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
    """Run one job under a heartbeat, updating the queue on success/failure."""
    stop = threading.Event()

    def _beat():
        while not stop.wait(HEARTBEAT_INTERVAL):
            try:
                job_queue.heartbeat(job["id"])
            except Exception:
                pass  # heartbeat failure shouldn't kill the job

    beat = threading.Thread(target=_beat, daemon=True)
    beat.start()
    try:
        _run_job(job)
        job_queue.mark_done(job["id"])
        print(f"  job {job['id']} ({job['kind']}, run {job['run_id']}) done")
    except Exception as e:
        traceback.print_exc()
        requeued = job_queue.mark_failed(
            job["id"], e, job["attempts"], job["max_attempts"])
        state = "requeued for retry" if requeued else "FAILED (out of attempts)"
        print(f"  job {job['id']} ({job['kind']}) errored: {e} — {state}")
    finally:
        stop.set()


def main():
    print(f"Worker {job_queue.WORKER_ID} starting.")
    # Startup orphan recovery: a previous worker may have died mid-job.
    try:
        n = job_queue.reclaim_orphans()
        if n:
            print(f"  reclaimed {n} orphaned job(s) on startup")
    except Exception as e:
        print(f"  orphan recovery failed on startup: {e}")

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
            print(f"Claimed job {job['id']}: {job['kind']} (run {job['run_id']}, "
                  f"attempt {job['attempts']}/{job['max_attempts']})")
            _process(job)
        except KeyboardInterrupt:
            print("\nWorker stopping (Ctrl+C).")
            break
        except Exception as e:
            # A failure in the loop itself (e.g. DB blip) — log and keep going.
            print(f"  worker loop error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()