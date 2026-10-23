"""Backlog G: the checkpoint erasure guard covers every BIGINT run id.

The trigger matched '^[0-9]{1,18}$', so a tombstoned run with a 19-digit id
(BIGINT max is 9223372036854775807) kept accepting checkpoint writes. A 19-digit
value ABOVE the BIGINT range must not raise inside the trigger either.
"""
import importlib

import pytest

pytestmark = [pytest.mark.db]

from database import get_connection

MAX_BIGINT = "9223372036854775807"


def _exec(q, args=()):
    with get_connection() as c:
        cur = c.cursor()
        cur.execute(q, args)
        return cur.fetchone() if cur.description else None


def _reinstall_with_migration():
    from checkpointing import setup_schema
    setup_schema()                                  # the checkpoint tables exist
    mig = importlib.import_module("migrations.0018_checkpoint_guard_bigint")
    with get_connection() as c:
        mig.upgrade(c.cursor())


def _checkpoint(thread, cid):
    _exec("DELETE FROM checkpoints WHERE thread_id = %s AND checkpoint_id = %s", (thread, cid))
    _exec("INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, "
          "metadata) VALUES (%s, '', %s, '{}', '{}')", (thread, cid))
    return _exec("SELECT count(*) FROM checkpoints WHERE thread_id = %s AND checkpoint_id = %s",
                 (thread, cid))[0]


def test_max_bigint_run_id_is_guarded():
    _reinstall_with_migration()
    _exec("INSERT INTO erasure_tombstones (run_id, reason) VALUES (%s, 'run_deleted') "
          "ON CONFLICT (run_id) DO NOTHING", (int(MAX_BIGINT),))
    assert _checkpoint(MAX_BIGINT, "g1") == 0


def test_overflowing_19_digit_thread_does_not_raise():
    _reinstall_with_migration()
    assert _checkpoint("9999999999999999999", "g2") == 1      # not a run id: kept
    assert _checkpoint("92233720368547758070", "g3") == 1     # 20 digits: kept
    _exec("DELETE FROM checkpoints WHERE checkpoint_id IN ('g2', 'g3')")
