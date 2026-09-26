"""
Create / upgrade the database schema.

There is exactly ONE way to build the schema: the migration runner. This file used
to hand-build a "complete" baseline that silently drifted from migrations 0002+
(ownership, job queue, error codes, leases, backoff, NOT NULL owners, ...). It is
now just an alias kept for muscle memory and old docs:

    python db_pg.py      ==      python migrate.py

Empty database -> migrations 0001..NNNN -> latest schema. Existing database -> only
the pending migrations are applied.
"""
from migrate import migrate

if __name__ == "__main__":
    migrate()