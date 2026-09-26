"""Per-model HNSW indexes on image_embeddings.

image_embeddings.embedding has no fixed dimension so several models (ViT-B/32
at 512, ViT-L/14 at 768, fine-tuned variants) can share the table. pgvector
can only index a typed column, so each model gets a partial index on a cast:

    CREATE INDEX ... USING hnsw ((embedding::vector(512)) vector_cosine_ops)
        WHERE model = 'openclip-vitb32'

Queries must repeat the exact same cast and WHERE clause for the planner to
use it; `image_distance_sql` is the single place that builds that expression.
"""

from __future__ import annotations

import re

from psycopg import sql

from api.db.session import Conn

_SAFE = re.compile(r"[^a-z0-9]+")


def index_name(model: str) -> str:
    return f"image_emb_hnsw_{_SAFE.sub('_', model.lower()).strip('_')}"[:63]


def ensure_image_index(conn: Conn, model: str, dim: int) -> None:
    conn.execute(
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {idx} ON image_embeddings USING hnsw "
            "((embedding::vector({dim})) vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 128) WHERE model = {model}"
        ).format(
            idx=sql.Identifier(index_name(model)),
            dim=sql.Literal(dim),
            model=sql.Literal(model),
        )
    )
    conn.commit()


def image_distance_sql(dim: int, alias: str = "e") -> str:
    """SQL fragment for cosine distance to query param %(qvec)s. `dim` is an int we control."""
    return f"({alias}.embedding::vector({int(dim)})) <=> %(qvec)s::vector({int(dim)})"
