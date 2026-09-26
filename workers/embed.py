"""Batch CLIP image embedding.

Reads the 512 px thumbnails (not originals): decoding is ~20x cheaper and CLIP
only sees 224 px anyway. Image loading runs in a thread pool one batch ahead
of the GPU so the model never waits on disk. Resumable: each run only embeds
photos that have no row for this model yet.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from api.config import get_settings
from api.db.session import get_conn
from api.db.util import vec
from api.db.vector_index import ensure_image_index
from api.ml.clip import get_clip
from api.paths import resolve

log = logging.getLogger(__name__)

PENDING_SQL = """
SELECT p.id, p.thumb_path FROM photos p
WHERE p.media_type = 'image' AND p.thumb_path IS NOT NULL AND p.error IS NULL
  AND NOT EXISTS (SELECT 1 FROM image_embeddings e WHERE e.photo_id = p.id AND e.model = %(model)s)
  {extra}
ORDER BY p.id
LIMIT %(limit)s
"""


def _load(path: str) -> Image.Image | None:
    try:
        with Image.open(resolve(path) or path) as im:
            return im.convert("RGB")
    except Exception as e:
        log.warning("cannot open thumbnail %s: %s", path, e)
        return None


def _batches(rows: list[dict], size: int) -> Iterator[list[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def pending(
    model: str, limit: int | None = None, ids: Iterable[uuid.UUID] | None = None
) -> list[dict]:
    extra = "AND p.id = ANY(%(ids)s)" if ids is not None else ""
    with get_conn() as conn:
        return conn.execute(
            PENDING_SQL.format(extra=extra),
            {"model": model, "limit": limit, "ids": list(ids) if ids is not None else None},
        ).fetchall()


def embed_pending(
    model: str | None = None,
    *,
    ids: Iterable[uuid.UUID] | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> int:
    s = get_settings()
    enc = get_clip(model)
    model_name = enc.spec.name
    batch_size = batch_size or s.embed_batch_size
    rows = pending(model_name, limit, ids)
    if not rows:
        return 0
    with get_conn() as conn:
        ensure_image_index(conn, model_name, enc.dim)

    done = 0
    with ThreadPoolExecutor(8) as pool:
        batches = list(_batches(rows, batch_size))
        # prefetch: images for batch i+1 decode while batch i is on the GPU
        next_imgs = pool.map(_load, [r["thumb_path"] for r in batches[0]])
        for bi, batch in enumerate(batches):
            imgs = list(next_imgs)
            if bi + 1 < len(batches):
                next_imgs = pool.map(_load, [r["thumb_path"] for r in batches[bi + 1]])
            ok = [(r, im) for r, im in zip(batch, imgs, strict=True) if im is not None]
            if not ok:
                continue
            vecs = enc.encode_images([im for _, im in ok])
            write_embeddings(model_name, [r["id"] for r, _ in ok], vecs)
            done += len(ok)
            if progress:
                progress("embed", done, len(rows))
    return done


def write_embeddings(model: str, ids: list[uuid.UUID], vecs: np.ndarray) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO image_embeddings (photo_id, model, embedding) VALUES (%s, %s, %s) "
                "ON CONFLICT (photo_id, model) DO UPDATE SET embedding = EXCLUDED.embedding, "
                "created_at = now()",
                [(pid, model, v) for pid, v in zip(ids, vecs, strict=True)],
            )
        conn.commit()


def load_embeddings(
    model: str, where: str = "", params: dict | None = None
) -> tuple[list[uuid.UUID], np.ndarray]:
    """All embeddings for a model as (ids, matrix). Used by dedupe, bursts, training."""
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT e.photo_id, e.embedding FROM image_embeddings e JOIN photos p ON p.id = e.photo_id "
            f"WHERE e.model = %(model)s {where} ORDER BY e.photo_id",
            {"model": model, **(params or {})},
        ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), np.float32)
    return [r["photo_id"] for r in rows], np.stack([vec(r["embedding"]) for r in rows])
