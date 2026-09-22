"""
Migration 0007 — enforce resumes.user_id / runs.user_id NOT NULL.

0002 added the ownership columns and set NOT NULL only when a user already existed
at migration time (so its backfill had an owner to assign). On a brand-new database
no user exists when 0002 runs, so it left the columns NULLABLE — and there they
stayed. The API always writes user_id, but the DATABASE never enforced it, so the
application rule ("every row has an owner") and the schema rule disagreed.

This migration closes that gap. The key point 0002 missed: SET NOT NULL doesn't need
a user to exist — it needs no NULL rows. On a fresh DB both tables are EMPTY, so
enforcement succeeds immediately. On an in-use DB, any pre-ownership NULL rows are
first backfilled to the lowest user id (the first/admin account, matching 0002).

If NULL rows remain AND there is no user to own them (owned data with no possible
owner — pathological), we FAIL LOUDLY rather than silently leaving the invariant
unenforced: the fix is to create an admin user or remove the orphan rows, then
re-run. Idempotent: SET NOT NULL on an already-NOT NULL column is a no-op.
"""


def upgrade(cur):
    # 1. Backfill any pre-ownership NULLs to the lowest user id, if a user exists.
    cur.execute("SELECT id FROM users ORDER BY id ASC LIMIT 1")
    row = cur.fetchone()
    if row:
        owner_id = row[0]
        cur.execute("UPDATE resumes SET user_id = %s WHERE user_id IS NULL", (owner_id,))
        cur.execute("UPDATE runs    SET user_id = %s WHERE user_id IS NULL", (owner_id,))

    # 2. Enforce NOT NULL wherever no NULLs remain (always true for an empty table).
    for table in ("resumes", "runs"):
        cur.execute(f"SELECT count(*) FROM {table} WHERE user_id IS NULL")
        remaining = cur.fetchone()[0]
        if remaining == 0:
            cur.execute(f"ALTER TABLE {table} ALTER COLUMN user_id SET NOT NULL")
        else:
            raise RuntimeError(
                f"Cannot enforce {table}.user_id NOT NULL: {remaining} row(s) have a "
                f"NULL owner and no user exists to assign them to. Create an admin user "
                f"(or delete the orphan rows), then re-run migrations."
            )