"""
Centralized PostgreSQL access — a single connection pool for the whole app.

Every module gets its connections from here (get_connection()), instead of each
opening its own psycopg2.connect. Connections are drawn from a shared, lazily
created ThreadedConnectionPool and REUSED, which is far cheaper than opening a
fresh TCP+auth connection on every query (the old pattern).

Drop-in compatibility with existing call sites: the whole codebase does
    conn = get_connection(); cur = conn.cursor(); ...; conn.commit(); conn.close()
With a raw pool, conn.close() would really close the socket and drain the pool.
So get_connection() hands back a THIN PROXY whose .close() RETURNS the connection
to the pool instead of closing it. Everything else (cursor, commit, rollback,
context-manager use) passes straight through to the real connection. No call site
has to change how it opens or closes a connection.

Also usable as a context manager for new code:
    with get_connection() as conn:
        with conn.cursor() as cur:
            ...
    # returned to the pool automatically
"""
import threading
import psycopg2
from psycopg2 import pool as _pgpool

from settings import settings

_POOL = None
_POOL_LOCK = threading.Lock()

# Pool sizing: min kept warm, max hard ceiling. FastAPI runs endpoints in a
# threadpool and background agent runs also hit the DB, so allow a healthy max.
_MIN_CONN = 1
_MAX_CONN = 20


def _get_pool():
    """Lazily create the process-wide pool (thread-safe, created once). DB settings
    are validated HERE, on first real use — importing a module must never require a
    live DB config (pure-logic code and tests import freely without one)."""
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                settings.validate_db()   # fail loudly only when we actually connect
                _POOL = _pgpool.ThreadedConnectionPool(
                    _MIN_CONN, _MAX_CONN, **settings.db_kwargs()
                )
    return _POOL


class _PooledConnection:
    """
    Transparent wrapper around a pooled psycopg2 connection.

    Delegates every attribute (cursor, commit, rollback, etc.) to the real
    connection, but overrides close() to RETURN the connection to the pool rather
    than closing it — so existing `conn.close()` call sites keep working while the
    underlying socket is reused. Also supports `with get_connection() as conn:`.
    """

    def __init__(self, pool, conn):
        self._pool = pool
        self._conn = conn
        self._returned = False

    def __getattr__(self, name):
        # Anything we don't override (cursor, commit, rollback, autocommit, ...)
        # goes straight to the wrapped connection.
        return getattr(self._conn, name)

    def close(self):
        """
        Return the connection to the pool (idempotent).

        CRITICAL for pooling: before returning, roll back any in-progress or
        ABORTED transaction. The whole codebase uses `conn = get_connection(); ...;
        conn.close()`, and when a query raises mid-transaction the caller's except
        block calls close() with the connection still in a failed state. psycopg2's
        putconn does NOT reset it, so without this rollback the next borrower would
        get a poisoned connection ("current transaction is aborted"). A committed or
        clean connection rolls back harmlessly (no-op).
        """
        if self._returned or self._conn is None:
            return
        try:
            # status != IDLE means there's an open/aborted transaction to clear.
            if self._conn.get_transaction_status() != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                self._conn.rollback()
        except Exception:
            # Connection is broken — discard it from the pool (close=True) rather
            # than returning a dead socket that would fail for the next borrower.
            try:
                self._pool.putconn(self._conn, close=True)
            finally:
                self._returned = True
            return
        self._pool.putconn(self._conn)
        self._returned = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # On error, roll back so a broken transaction isn't returned to the pool
        # in a dirty state; then always return the connection.
        try:
            if exc_type is not None:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
        finally:
            self.close()
        return False


def get_connection():
    """
    Borrow a connection from the shared pool. Use exactly like before:
        conn = get_connection()
        cur = conn.cursor()
        ...
        conn.commit()
        conn.close()   # returns to the pool (does NOT really close)
    or as a context manager for new code.
    """
    pool = _get_pool()
    raw = pool.getconn()
    return _PooledConnection(pool, raw)


def close_pool():
    """Close every pooled connection. For clean shutdown or tests; the app rarely
    needs this since the pool lives for the process lifetime."""
    global _POOL
    if _POOL is not None:
        _POOL.closeall()
        _POOL = None