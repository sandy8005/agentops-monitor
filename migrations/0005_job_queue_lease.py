"""
Migration 0005 — job_queue.lease_token (fencing token for orphan recovery).

Orphan recovery can return a job that a worker is STILL running (its heartbeat
just failed transiently for longer than ORPHAN_AFTER) to 'queued', where a second
worker then claims it. Without a fencing token, the original worker could still
mark that job done/failed by id alone, corrupting the queue record and masking the
duplicate execution.

lease_token is a per-claim UUID: claim_next() stamps a fresh token, and every
subsequent mutation of that row (heartbeat, mark_done, mark_failed) is guarded by
`AND status = 'running' AND lease_token = <token>`. Once orphan recovery clears the
token (requeue) and another worker claims a NEW token, the original worker's writes
match zero rows — it has lost permission to mutate the record.

NULL while a job is 'queued' (no owner yet); set on claim; cleared on requeue.
Idempotent: ADD COLUMN IF NOT EXISTS.
"""


def upgrade(cur):
    cur.execute("ALTER TABLE job_queue ADD COLUMN IF NOT EXISTS lease_token TEXT")