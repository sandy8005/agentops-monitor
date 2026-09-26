"""
Single source of "now" for the application.

All timestamp columns are TIMESTAMPTZ (migration 0008), so the application writes
timezone-AWARE UTC datetimes. Naive datetime.now() values mean "whatever the server's
local zone is", which silently breaks once the API, worker, and database don't share
a time zone — and aware values read back from TIMESTAMPTZ can't be compared with
naive ones at all (TypeError).
"""
from datetime import datetime, timezone


def utcnow():
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)