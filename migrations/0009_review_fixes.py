"""
Migration 0009 — review fixes: cost provenance, privacy-safe parse cache, queue
vocabulary, provider naming, and guaranteed integrity constraints. Idempotent.

  llm_calls.logical_call_id / pricing_version
        Group the HTTP attempts of one logical LLM call, and record WHICH price
        table produced cost_usd (costs are paid-tier ESTIMATES; see pricing.py).

  parsed_resume_cache: (content_hash, cache_version)
        The old key was hash(resume text + parser/schema/model version), so erasing a
        resume could only find the CURRENT version's row — rows parsed under older
        versions (derived personal data: skills, education, experience) survived
        deletion. The cache is now keyed by a STABLE content_hash = sha256(text) plus
        a separate cache_version, UNIQUE(content_hash, cache_version), so erasure is
        simply DELETE ... WHERE content_hash = ?, whatever versions ever existed.
        Legacy rows that can be mapped to a stored resume under the current version
        are converted; every other legacy row is unreachable anyway and is DELETED
        (it would otherwise be undeletable personal data).

  job_queue.kind CHECK
        Only kinds the worker can execute may be enqueued — a typo can no longer
        create an unprocessable durable job.

  job_queue.orphan_reclaim_count / last_error_at
        Structured orphan bookkeeping (replaces appending " [reclaimed orphan]" to
        last_error on every reclaim).

  resumes.original_chars / truncated
        An oversized PDF extraction is capped at storage time; the row now records
        that it happened and how much text there originally was.

  runs.trace_purged_at
        When the retention scrubber last processed the run, so each sweep only
        touches new expirations (eligibility itself no longer depends on which
        payload columns happen to still be populated).

  job_postings.source 'live' -> 'remotive'
        Name the provider, consistent with job_searches.source and 'adzuna'.

  0008 uniqueness constraints
        0008 used to log-and-skip a UNIQUE constraint that couldn't be added and still
        record itself as applied. Any database where that happened is repaired here:
        duplicates are remediated and the constraint added — or the migration fails.
"""
import hashlib
import importlib.util
import os

QUEUE_KINDS = ("start_run", "resume_run")


def _load_0008():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "0008_integrity_and_observability.py")
    spec = importlib.util.spec_from_file_location("_mig0008", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _column_exists(cur, table, col):
    cur.execute("SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s AND column_name = %s",
                (table, col))
    return cur.fetchone() is not None


def _migrate_parse_cache(cur):
    cur.execute("ALTER TABLE parsed_resume_cache ADD COLUMN IF NOT EXISTS content_hash TEXT")
    if _column_exists(cur, "parsed_resume_cache", "resume_hash"):
        # Map legacy rows written under the CURRENT cache version back to their
        # resume text (legacy key = sha256(f"{text}|{version}")[:16]).
        from cache_version import parse_cache_version
        version = parse_cache_version()
        cur.execute("SELECT resume_text FROM resumes WHERE resume_text IS NOT NULL "
                    "AND resume_text <> '[erased]'")
        for (text,) in cur.fetchall():
            legacy = hashlib.sha256(f"{text}|{version}".encode("utf-8")).hexdigest()[:16]
            content = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cur.execute("UPDATE parsed_resume_cache SET content_hash = %s, cache_version = %s "
                        "WHERE resume_hash = %s AND content_hash IS NULL",
                        (content, version, legacy))
        # Everything else is unreachable by the app AND undeletable by erase_resume.
        cur.execute("DELETE FROM parsed_resume_cache WHERE content_hash IS NULL")
        if cur.rowcount:
            print(f"  parsed_resume_cache: deleted {cur.rowcount} unmappable legacy row(s)")
        cur.execute("ALTER TABLE parsed_resume_cache DROP CONSTRAINT IF EXISTS "
                    "parsed_resume_cache_pkey")
        cur.execute("ALTER TABLE parsed_resume_cache DROP COLUMN resume_hash")
    cur.execute("DELETE FROM parsed_resume_cache WHERE content_hash IS NULL OR cache_version IS NULL")
    cur.execute("ALTER TABLE parsed_resume_cache ALTER COLUMN content_hash SET NOT NULL")
    cur.execute("ALTER TABLE parsed_resume_cache ALTER COLUMN cache_version SET NOT NULL")
    if not _column_exists(cur, "parsed_resume_cache", "id"):
        cur.execute("ALTER TABLE parsed_resume_cache ADD COLUMN id BIGSERIAL PRIMARY KEY")
    mig8 = _load_0008()
    mig8._DEDUP_KEEP.setdefault("parsed_resume_cache_content_version_uniq", "min")
    mig8._add_unique(cur, "parsed_resume_cache", "parsed_resume_cache_content_version_uniq",
                     "content_hash, cache_version")


def upgrade(cur):
    mig8 = _load_0008()

    # ------------------------------------------------------------ columns --
    for table, col, typ in [
        ("llm_calls", "logical_call_id", "TEXT"),
        ("llm_calls", "pricing_version", "TEXT"),
        ("job_queue", "orphan_reclaim_count", "INTEGER NOT NULL DEFAULT 0"),
        ("job_queue", "last_error_at", "TIMESTAMPTZ"),
        ("resumes", "original_chars", "INTEGER"),
        ("resumes", "truncated", "BOOLEAN NOT NULL DEFAULT FALSE"),
        ("runs", "trace_purged_at", "TIMESTAMPTZ"),
    ]:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}")
    cur.execute("CREATE INDEX IF NOT EXISTS llm_calls_logical_idx ON llm_calls (logical_call_id)")

    # Legacy orphan markers appended to last_error: count them once, then strip them.
    cur.execute("""
        UPDATE job_queue
        SET orphan_reclaim_count = orphan_reclaim_count
                + (length(last_error) - length(replace(last_error, '[reclaimed orphan]', '')))
                  / length('[reclaimed orphan]'),
            last_error = NULLIF(btrim(replace(last_error, '[reclaimed orphan]', '')), '')
        WHERE last_error LIKE '%%[reclaimed orphan]%%'
    """)

    # --------------------------------------------------------- parse cache --
    _migrate_parse_cache(cur)

    # ------------------------------------------------------- provider names --
    cur.execute("UPDATE job_postings SET source = 'remotive' WHERE source = 'live'")

    # --------------------------------------------------------- constraints --
    mig8._add_constraint(cur, "job_queue", "job_queue_kind_chk",
                         f"CHECK (kind IN {mig8._in(QUEUE_KINDS)})")

    # Guarantee the 0008 uniqueness invariants (repairs DBs where 0008 skipped them).
    mig8._add_unique(cur, "run_rankings", "run_rankings_run_pos_uniq", "run_id, rank_position")
    mig8._add_unique(cur, "run_rankings", "run_rankings_run_job_uniq", "run_id, job_id")
    mig8._add_unique(cur, "run_advice", "run_advice_run_job_uniq", "run_id, job_id")