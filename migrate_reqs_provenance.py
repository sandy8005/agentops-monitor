"""
Migration: requirements-cache provenance.

Records HOW each cached requirements row was produced, so a cache hit can be
traced and trusted (or refreshed) appropriately:
  - extraction_method : 'llm' (Gemini extraction) or 'rule_based' (no-LLM fallback)
  - source_model      : the model/version string in use when it was cached
                        (e.g. the reqs cache_version), so rows are attributable to
                        a specific extractor generation.

Also ensures created_at exists (the original table had it; this is idempotent).
The cache_version column is added by migrate_cache_version.py; this migration does
not touch it.

Idempotent: safe to run more than once. Existing rows get NULL provenance — that's
fine, since the REQS_VERSION bump means the new (title+description) key won't hit
them anyway; they simply age out.
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()
cur.execute("ALTER TABLE job_reqs_cache ADD COLUMN IF NOT EXISTS extraction_method TEXT")
cur.execute("ALTER TABLE job_reqs_cache ADD COLUMN IF NOT EXISTS source_model TEXT")
cur.execute("ALTER TABLE job_reqs_cache ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()")
conn.commit()
conn.close()
print("Migration complete: job_reqs_cache.extraction_method + source_model added")