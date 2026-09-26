"""Tiny forward-only migration runner: applies api/db/migrations/NNN_*.sql in order."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def pending(conn: psycopg.Connection[Any]) -> list[Path]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    done = {r["version"] if isinstance(r, dict) else r[0] for r in rows}
    return [p for p in sorted(MIGRATIONS_DIR.glob("*.sql")) if p.stem not in done]


def migrate(url: str) -> list[str]:
    """Apply pending migrations; each file runs in its own transaction."""
    applied: list[str] = []
    with psycopg.connect(url, autocommit=False) as conn:
        todo = pending(conn)
        conn.commit()
        for path in todo:
            with conn.transaction():
                conn.execute(path.read_text())
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
            applied.append(path.stem)
    # Default image model index. Other models get theirs when first embedded.
    from api.db.vector_index import ensure_image_index
    from api.ml.registry import get_image_model_spec

    from .session import connect

    with connect(url) as conn:
        spec = get_image_model_spec()
        ensure_image_index(conn, spec.name, spec.dim)
    return applied
