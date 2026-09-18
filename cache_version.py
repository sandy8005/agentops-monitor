"""
Cache versioning. Bump the relevant version when you change something that would
make old cached results invalid, so stale entries become unreachable (via the
versioned key) instead of being served forever. The cache_version STRING is also
stored on each row so you can query/prune by version.

- PARSER_VERSION: bump when the resume-parsing PROMPT changes.
- REQS_VERSION:   bump when the requirements-extraction PROMPT changes.
- SCHEMA_VERSION: bump when Pydantic validation rules change in a way that affects
                  what a valid parsed result looks like.
- model:          the Gemini model in use is NOT duplicated here — it comes from
                  the single source of truth (settings.gemini_model), so the cache
                  key's model tag can never drift from the model llm.py actually
                  calls. Changing the model (via GEMINI_MODEL) therefore also
                  invalidates old cache entries, which is correct: a different
                  model can produce different output for the same input.
"""
from settings import settings

PARSER_VERSION = "1"
REQS_VERSION = "2"   # bumped: cache key now includes TITLE + description (was desc-only)
SCHEMA_VERSION = "1"


def model_version():
    """The model name, from the single source of truth (settings)."""
    return settings.gemini_model


def parse_cache_version():
    """Composite version string for the resume-parse cache."""
    return f"parser={PARSER_VERSION};schema={SCHEMA_VERSION};model={model_version()}"


def reqs_cache_version():
    """Composite version string for the requirements cache."""
    return f"reqs={REQS_VERSION};schema={SCHEMA_VERSION};model={model_version()}"