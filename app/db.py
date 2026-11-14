"""Postgres connection pool + small helpers shared by the API and offline pipelines."""
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

_pool: ConnectionPool | None = None


def init_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        s = get_settings()
        # open(wait=False): the API can boot (and /health respond) even if the DB is down.
        _pool = ConnectionPool(
            s.database_url,
            min_size=1,
            max_size=s.db_pool_max,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        _pool.open(wait=False)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def get_conn() -> Iterator[psycopg.Connection]:
    """FastAPI dependency: one pooled connection per request, committed on success."""
    with init_pool().connection() as conn:
        yield conn


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """Standalone connection for scripts/pipelines."""
    with psycopg.connect(get_settings().database_url, row_factory=dict_row) as conn:
        yield conn


def to_vector(embedding: Sequence[float]) -> str:
    """pgvector text literal, e.g. '[0.1,0.2,...]'. Use with a ::vector cast."""
    return "[" + ",".join(f"{float(x):.7g}" for x in embedding) + "]"
