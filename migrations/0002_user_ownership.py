"""
Migration 0002 — data ownership (authorization).

Authentication proved WHO you are; it did not scope WHAT you can access. Any
logged-in user could read/mutate another user's resumes and runs by guessing an
integer id (IDOR). This adds an owner to the two ROOT entities:

    resumes.user_id  -> users(id)
    runs.user_id     -> users(id)

Child tables (steps, llm_calls, tool_calls, evaluations, run_rankings, run_advice)
are reached only through run_id, so scoping the runs query by owner protects them
transitively — no per-child user_id needed.

Ownership is enforced in the API: every query is scoped `... AND user_id = %s`.

Backfill: existing rows predate ownership. We assign them to the LOWEST user id
(the first/admin account) so historical data stays visible to that owner rather
than being orphaned. If there are NO users yet, the columns are left nullable and
NOT NULL is deferred.

Idempotent: ADD COLUMN IF NOT EXISTS + guarded constraint add.
"""


def upgrade(cur):
    # 1. Add the ownership columns (nullable for now so the backfill can run).
    cur.execute("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS user_id INTEGER")
    cur.execute("ALTER TABLE runs    ADD COLUMN IF NOT EXISTS user_id INTEGER")

    # 2. Foreign keys to users(id). Guarded so re-running doesn't error on a dup.
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'resumes_user_id_fkey') THEN
                ALTER TABLE resumes
                    ADD CONSTRAINT resumes_user_id_fkey
                    FOREIGN KEY (user_id) REFERENCES users(id);
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'runs_user_id_fkey') THEN
                ALTER TABLE runs
                    ADD CONSTRAINT runs_user_id_fkey
                    FOREIGN KEY (user_id) REFERENCES users(id);
            END IF;
        END $$;
    """)

    # 3. Indexes for the per-owner scoping queries.
    cur.execute("CREATE INDEX IF NOT EXISTS resumes_user_id_idx ON resumes (user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS runs_user_id_idx    ON runs (user_id)")

    # 4. Backfill existing rows to the lowest user id (first/admin), if any user exists.
    cur.execute("SELECT id FROM users ORDER BY id ASC LIMIT 1")
    row = cur.fetchone()
    if row:
        owner_id = row[0]
        cur.execute("UPDATE resumes SET user_id = %s WHERE user_id IS NULL", (owner_id,))
        cur.execute("UPDATE runs    SET user_id = %s WHERE user_id IS NULL", (owner_id,))

        # 5. Enforce NOT NULL now that no NULLs remain.
        cur.execute("SELECT count(*) FROM resumes WHERE user_id IS NULL")
        resumes_nulls = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM runs WHERE user_id IS NULL")
        runs_nulls = cur.fetchone()[0]
        if resumes_nulls == 0:
            cur.execute("ALTER TABLE resumes ALTER COLUMN user_id SET NOT NULL")
        if runs_nulls == 0:
            cur.execute("ALTER TABLE runs ALTER COLUMN user_id SET NOT NULL")
    # NOTE: if there were no users yet, columns stay nullable. In normal operation
    # the API always writes user_id, so new rows are never NULL.