"""
Cache versioning. Bump the relevant version when you change something that would
make old cached results invalid, so stale entries become unreachable (via the
versioned key) instead of being served forever. The cache_version STRING is also
stored on each row so you can query/prune by version.

- PARSER_VERSION: bump when the resume-parsing PROMPT changes.
- REQS_VERSION:   bump when the requirements-extraction PROMPT changes.
- SCHEMA_VERSION: bump when Pydantic validation rules change in a way that affects
                  what a valid parsed result looks like.
- MODEL_VERSION:  the Gemini model string in use (keep in sync with llm.py).
"""
PARSER_VERSION = "1"
REQS_VERSION = "1"
SCHEMA_VERSION = "1"
MODEL_VERSION = "gemini-flash-latest"


def parse_cache_version():
    """Composite version string for the resume-parse cache."""
    return f"parser={PARSER_VERSION};schema={SCHEMA_VERSION};model={MODEL_VERSION}"


def reqs_cache_version():
    """Composite version string for the requirements cache."""
    return f"reqs={REQS_VERSION};schema={SCHEMA_VERSION};model={MODEL_VERSION}"