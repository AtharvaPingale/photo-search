"""Redis + RQ job queue.

Two queues:
  ingest  CPU work: scanning, EXIF, thumbnails, video frame extraction.
  gpu     model work, in batches of photo ids (one job = up to BATCH photos),
          so each job amortises the forward-pass batching and the worker's
          already-loaded models.

GPU workers run as RQ SimpleWorker (no fork per job): forking would reload
CLIP / Florence / InsightFace for every job.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from api.config import get_settings

log = logging.getLogger(__name__)
BATCH = 256


def redis_conn():
    from redis import Redis

    return Redis.from_url(get_settings().redis_url)


def queues(conn=None):
    from rq import Queue

    conn = conn or redis_conn()
    return Queue("ingest", connection=conn, default_timeout=6 * 3600), Queue(
        "gpu", connection=conn, default_timeout=6 * 3600
    )


# ------------------------------------------------------------------ jobs


def job_scan(paths: list[str] | None, captions: bool, ocr: bool, faces: bool) -> dict[str, Any]:
    from workers.ingest import scan
    from workers.video import index_videos

    stats = scan([Path(p) for p in paths] if paths else None)
    frames = index_videos() if get_settings().video_enabled else 0
    enqueued = enqueue_gpu_work(captions=captions, ocr=ocr, faces=faces)
    return {"scan": stats.summary(), "video_frames": frames, "enqueued": enqueued}


def job_embed(ids: list[str]) -> int:
    from workers.embed import embed_pending

    return embed_pending(ids=[uuid.UUID(i) for i in ids])


def job_caption(ids: list[str]) -> int:
    from workers.caption import caption_pending

    return caption_pending(ids=[uuid.UUID(i) for i in ids])


def job_ocr(ids: list[str]) -> dict[str, int]:
    from workers.ocr import ocr_pending

    return ocr_pending(ids=[uuid.UUID(i) for i in ids])


def job_faces(ids: list[str]) -> int:
    from workers.faces import detect_pending

    return detect_pending(ids=[uuid.UUID(i) for i in ids])


def job_cluster_faces() -> dict[str, int]:
    from workers.faces import cluster_faces

    return cluster_faces()


# ------------------------------------------------------------------ enqueueing


def _pending_ids(sql: str, params: dict | None = None) -> list[str]:
    from api.db.session import get_conn

    with get_conn() as conn:
        return [str(r["id"]) for r in conn.execute(sql, params or {}).fetchall()]


def _chunks(xs: list[str], n: int = BATCH) -> Iterable[list[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def enqueue_gpu_work(
    *, captions: bool = True, ocr: bool = True, faces: bool = False
) -> dict[str, int]:
    """Enqueue batches for everything missing. Embeddings go first; caption and OCR
    jobs depend on the embed job for the same batch (OCR's gate reads the embedding)."""
    from api.ml.registry import get_image_model_spec

    _, gpu = queues()
    model = get_image_model_spec().name
    ids = _pending_ids(
        "SELECT p.id FROM photos p WHERE p.media_type = 'image' AND p.error IS NULL AND p.thumb_path IS NOT NULL "
        "AND (NOT EXISTS (SELECT 1 FROM image_embeddings e WHERE e.photo_id = p.id AND e.model = %(m)s) "
        "  OR NOT EXISTS (SELECT 1 FROM captions c WHERE c.photo_id = p.id) "
        "  OR NOT EXISTS (SELECT 1 FROM ocr_text o WHERE o.photo_id = p.id) "
        "  OR p.faces_indexed_at IS NULL) ORDER BY p.id",
        {"m": model},
    )
    counts = {"embed": 0, "caption": 0, "ocr": 0, "faces": 0}
    s = get_settings()
    for chunk in _chunks(ids):
        ej = gpu.enqueue(job_embed, chunk, job_timeout=3600)
        counts["embed"] += 1
        if captions and s.caption_backend != "none":
            gpu.enqueue(job_caption, chunk, depends_on=ej, job_timeout=3 * 3600)
            counts["caption"] += 1
        if ocr and s.ocr_backend != "none":
            gpu.enqueue(job_ocr, chunk, depends_on=ej, job_timeout=3 * 3600)
            counts["ocr"] += 1
        if faces and s.face_roots:
            gpu.enqueue(job_faces, chunk, job_timeout=3 * 3600)
            counts["faces"] += 1
    if faces and s.face_roots and counts["faces"]:
        gpu.enqueue(job_cluster_faces, at_front=False)
    return counts


def enqueue_pipeline(
    paths: Iterable[Path] | None, *, captions: bool, ocr: bool, faces: bool
) -> dict[str, Any]:
    ingest, _ = queues()
    job = ingest.enqueue(job_scan, [str(p) for p in paths] if paths else None, captions, ocr, faces)
    return {"scan_job": job.id}


def enqueue_ids(
    ids: list[uuid.UUID], *, captions: bool = True, ocr: bool = True, faces: bool = True
) -> None:
    """Used by the watcher: a handful of new photos -> one small batch per stage."""
    _, gpu = queues()
    s = get_settings()
    chunk = [str(i) for i in ids]
    ej = gpu.enqueue(job_embed, chunk)
    if captions and s.caption_backend != "none":
        gpu.enqueue(job_caption, chunk, depends_on=ej)
    if ocr and s.ocr_backend != "none":
        gpu.enqueue(job_ocr, chunk, depends_on=ej)
    if faces and s.face_roots:
        gpu.enqueue(job_faces, chunk)


def run_worker(queue_names: list[str]) -> None:
    from rq import SimpleWorker, Worker

    conn = redis_conn()
    from rq import Queue

    qs = [Queue(n, connection=conn) for n in queue_names]
    cls = SimpleWorker if "gpu" in queue_names else Worker
    cls(qs, connection=conn).work(with_scheduler=False)
