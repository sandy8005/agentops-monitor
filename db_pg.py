"""
Create / upgrade the database schema.

There is exactly ONE way to build the schema: the migration runner. This file used
to hand-build a "complete" baseline that silently drifted from migrations 0002+
(ownership, job queue, error codes, leases, backoff, NOT NULL owners, ...). It is
now an alias kept for muscle memory and old docs, and it calls the SAME entrypoint
as migrate.py, flags included:

    python db_pg.py [--status | --skip-checkpointer]
        ==
    python migrate.py [--status | --skip-checkpointer]

Empty database -> migrations 0001..NNNN -> latest schema, then LangGraph's
checkpoint tables and their erasure guards. Existing database -> only the pending
migrations are applied.
"""
import sys

from migrate import main

if __name__ == "__main__":
    sys.exit(main())