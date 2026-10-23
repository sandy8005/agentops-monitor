"""
Migration 0017 — work-mode labels on persisted rankings. Idempotent.

  run_rankings.work_mode            the posting's STATED work mode, or 'unknown'
  run_rankings.work_mode_verified   FALSE when the posting states no work mode:
                                    a remote search keeps such postings (providers
                                    often omit the field) but must never present
                                    them as confirmed remote matches.
                                    NULL for rows ranked before this migration.
"""


def upgrade(cur):
    cur.execute("ALTER TABLE run_rankings ADD COLUMN IF NOT EXISTS work_mode TEXT")
    cur.execute("ALTER TABLE run_rankings ADD COLUMN IF NOT EXISTS work_mode_verified BOOLEAN")
