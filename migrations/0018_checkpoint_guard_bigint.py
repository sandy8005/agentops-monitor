"""
Migration 0018 — the checkpoint erasure guard covers every BIGINT run id. Idempotent.

  erasure_guard_checkpoint()
        Matched thread ids with '^[0-9]{1,18}$', so a tombstoned run whose id has
        19 digits (BIGINT allows up to 9223372036854775807) could still have
        checkpoints written. It now accepts up to 19 digits and compares the value
        as NUMERIC before casting, so an id above the BIGINT range is ignored
        instead of raising inside the trigger. Reinstalled here for databases
        whose checkpoint tables already exist; fresh databases get it from
        checkpointing.setup_schema() (erasure_guards.install_checkpoint_guards).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erasure_guards import install_checkpoint_guards  # noqa: E402


def upgrade(cur):
    install_checkpoint_guards(cur)
