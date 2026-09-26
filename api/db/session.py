"""Connection handling.

The search core is synchronous: every request does one text-encoder forward
pass plus one or two SQL queries, and FastAPI runs sync endpoints on its
threadpool. One sync pool is shared by the API, workers, eval runner and agent,
so they all hit exactly the same code path.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from api.config import get_settings

Conn = psycopg.Connection[DictRow]

_pool: ConnectionPool[Conn] | None = None
_pool_url: str | None = None
_lock = threading.Lock()


def _configure(conn: Conn) -> None:
    conn.execute("SET TimeZone = 'UTC'")
    register_vector(conn)
    conn.commit()


def connect(url: str | None = None, *, vector: bool = True) -> Conn:
    """A standalone connection (migrations, scripts). Rows come back as dicts."""
    conn = psycopg.connect(url or get_settings().database_url, row_factory=dict_row)
    conn.execute("SET TimeZone = 'UTC'")
    if vector:
        register_vector(conn)
    conn.commit()
    return conn


def get_pool() -> ConnectionPool[Conn]:
    global _pool, _pool_url
    url = get_settings().database_url
    with _lock:
        if _pool is None or _pool_url != url:
            if _pool is not None:
                _pool.close()
            _pool = ConnectionPool(
                url,
                connection_class=Conn,
                min_size=1,
                max_size=16,
                kwargs={"row_factory": dict_row},
                configure=_configure,
                open=True,
            )
            _pool_url = url
        return _pool


def close_pool() -> None:
    global _pool, _pool_url
    with _lock:
        if _pool is not None:
            _pool.close()
        _pool = None
        _pool_url = None


@contextmanager
def get_conn() -> Iterator[Conn]:
    with get_pool().connection() as conn:
        yield conn


def one(conn: Conn, query: str, params: Any = None) -> dict[str, Any]:
    """The single row a query must return (aggregates, lookups by primary key)."""
    row = conn.execute(query, params).fetchone()
    if row is None:
        raise LookupError(f"query returned no rows: {query[:80]}")
    return row
