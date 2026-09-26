"""The indexing pipeline, run inline (no Redis): scan -> video frames -> embed -> caption -> OCR -> faces.

Every stage only processes what's missing, so the pipeline is safe to re-run
and to interrupt.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from api.config import get_settings


def run_pipeline(
    paths: Iterable[Path] | None = None,
    *,
    captions: bool = True,
    ocr: bool = True,
    faces: bool = False,
    ids: list[uuid.UUID] | None = None,
    scan: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    from workers import embed, video
    from workers import ingest as ing

    s = get_settings()
    out: dict[str, Any] = {}
    timings: dict[str, float] = {}

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        t = time.perf_counter()
        r = fn()
        timings[name] = round(time.perf_counter() - t, 2)
        out[name] = r
        return r

    if scan:
        stats = timed("scan", lambda: ing.scan(paths, progress=progress))
        out["scan"] = stats.summary()
    if s.video_enabled:
        timed("video_frames", lambda: video.index_videos())
    timed("embed", lambda: embed.embed_pending(ids=ids, progress=progress))
    if captions and s.caption_backend != "none":
        from workers.caption import caption_pending

        timed("caption", lambda: caption_pending(ids=ids, progress=progress))
    if ocr and s.ocr_backend != "none":
        from workers.ocr import ocr_pending

        timed("ocr", lambda: ocr_pending(ids=ids, progress=progress))
    if faces and s.face_roots:
        from workers.faces import cluster_faces, detect_pending

        timed("faces", lambda: detect_pending(ids=ids, progress=progress))
        timed("face_clusters", cluster_faces)
    out["timings_s"] = timings
    return out
