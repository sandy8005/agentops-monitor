"""
LangGraph Postgres checkpointer — construction, hardening, and schema setup.

Hardening (deserialization): checkpoint blobs are MessagePack. With LangGraph's
default permissive serializer, a type reference stored in a checkpoint is IMPORTED
and CALLED on load — so anyone able to write to the checkpoint tables could get code
execution in the worker. Following the package's own guidance, the checkpointer is
built with an explicit STRICT serializer (allowed_msgpack_modules=None: only
LangGraph's built-in SAFE_MSGPACK_TYPES), and LANGGRAPH_STRICT_MSGPACK=true is also
set by default (settings.py) as defense in depth. This app's graph state is a flat,
JSON-serializable dict, so it needs no custom types at all.

Schema setup: PostgresSaver.setup() creates/migrates LangGraph's own tables. It is
DDL and belongs to deployment, not to every job execution — it now runs once from
`python migrate.py` (and once at worker startup as a safety net), never per run.
"""
from contextlib import contextmanager

from logging_config import get_logger

log = get_logger(__name__)


def db_uri():
    """libpq conninfo for psycopg 3, built with make_conninfo (correct quoting of
    passwords with spaces/quotes), including connect_timeout and TCP keepalives."""
    from psycopg.conninfo import make_conninfo
    from settings import settings
    p = {k: str(v) for k, v in settings.db_kwargs().items() if v is not None}
    return make_conninfo(**p)


def make_serde():
    """Strict JSON-plus serializer: msgpack deserialization limited to LangGraph's
    SAFE_MSGPACK_TYPES allowlist (allowed_msgpack_modules=None), no pickle fallback."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    return JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=None)


@contextmanager
def open_checkpointer():
    """A PostgresSaver on its own psycopg 3 connection, with the strict serializer.
    Mirrors PostgresSaver.from_conn_string (autocommit, no prepared statements,
    dict rows) but lets us pass `serde`."""
    from psycopg import Connection
    from psycopg.rows import dict_row
    from langgraph.checkpoint.postgres import PostgresSaver
    with Connection.connect(db_uri(), autocommit=True, prepare_threshold=0,
                            row_factory=dict_row) as conn:
        yield PostgresSaver(conn, serde=make_serde())


def setup_schema():
    """Create / migrate LangGraph's checkpoint tables. Idempotent. Deployment-time.

    Serialized across processes with the same PostgreSQL advisory lock as the app
    migrations (migrate.schema_lock): two replicas deploying at once would
    otherwise both see a checkpoint migration as pending and race its DDL."""
    from migrate import schema_lock
    with schema_lock():
        with open_checkpointer() as cp:
            cp.setup()
    log.info("langgraph checkpoint schema is up to date")