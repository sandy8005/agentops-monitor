"""
The active runtime surface stays small: retired helpers that wrote run state
WITHOUT the execution-generation fence or cancellation precedence must not come
back, and superseded database-mutating scripts are not shipped.
"""
import os

import llm
import worker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_unfenced_run_writers_are_gone():
    # Each of these updated `runs` directly, bypassing agent_store's fenced,
    # cancel-aware finalize(): a stray call could overwrite a newer execution's
    # outcome or a user's cancellation.
    for name in ("finish_run", "request_cancel", "set_stop_reason",
                 "set_evaluation_status", "fake_llm"):
        assert not hasattr(llm, name), f"llm.{name} was retired"
    assert not hasattr(worker, "_finalize_run_failed_if_active")


def test_legacy_migration_scripts_are_not_shipped():
    assert not os.path.exists(os.path.join(ROOT, "scripts", "legacy_migrations"))