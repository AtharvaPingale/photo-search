from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from api.config import get_settings
from api.db.session import get_conn
from api.paths import resolve
from api.search.filters import SearchFilters, to_sql
from api.urls import display_url, original_url, thumb_url

router = APIRouter(tags=["photos"])

_BROWSER_OK = {"jpeg", "png", "webp", "gif", "bmp"}


def _photo(conn, photo_id: uuid.UUID) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM photos WHERE id = %s", (photo_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "photo not found")
    return row


@router.get("/photos/{photo_id}")
def photo_details(photo_id: uuid.UUID) -> dict[str, Any]:
    with get_conn() as conn:
        p = _photo(conn, photo_id)
        p.pop("keywords_tsv", None)
        captions = conn.execute(
            "SELECT model, caption FROM captions WHERE photo_id = %s ORDER BY created_at DESC",
            (photo_id,),
        ).fetchall()
        ocr = conn.execute(
            "SELECT text, gate_score FROM ocr_text WHERE photo_id = %s", (photo_id,)
        ).fetchone()
        faces = conn.execute(
            "SELECT f.id, f.bbox, f.cluster_id, pe.name FROM faces f "
            "LEFT JOIN people pe ON pe.cluster_id = f.cluster_id WHERE f.photo_id = %s",
            (photo_id,),
        ).fetchall()
        groups = conn.execute(
            "SELECT kind, group_id, is_best FROM photo_groups WHERE photo_id = %s", (photo_id,)
        ).fetchall()
        albums = conn.execute(
            "SELECT a.id, a.title FROM albums a JOIN album_photos ap ON ap.album_id = a.id "
            "WHERE ap.photo_id = %s",
            (photo_id,),
        ).fetchall()
        models = [
            r["model"]
            for r in conn.execute(
                "SELECT model FROM image_embeddings WHERE photo_id = %s", (photo_id,)
            )
        ]
    return {
        **p,
        "thumb_url": thumb_url(photo_id, p["file_hash"]),
        "display_url": display_url(photo_id, p["file_hash"]),
        "original_url": original_url(photo_id),
        "download_url": original_url(photo_id) + "?download=1",
        "captions": captions,
        "ocr_text": ocr["text"] if ocr else None,
        "faces": faces,
        "groups": groups,
        "albums": albums,
        "embedding_models": models,
    }


IMMUTABLE = {"Cache-Control": "private, max-age=31536000, immutable"}
DAY = {"Cache-Control": "private, max-age=86400"}
SMALL = 256
DISPLAY_SIDE = 2048


def _cache_headers(versioned: bool) -> dict[str, str]:
    return IMMUTABLE if versioned else DAY


def _small_thumb(photo_id: uuid.UUID, src: Path) -> Path:
    """256 px WebP made from the 512 px JPEG on first request, then cached.
    ~6-12 KB each, so a screenful of results loads fast on mobile data."""
    from PIL import Image

    s = str(photo_id)
    out = get_settings().small_thumbs_dir / s[:2] / f"{s}.webp"
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as f:
            im = f.convert("RGB")
            im.thumbnail((SMALL, SMALL), Image.Resampling.LANCZOS)
            tmp = out.with_suffix(".tmp")
            im.save(tmp, "WEBP", quality=72, method=4)
            tmp.replace(out)
    return out


@router.get("/photos/{photo_id}/thumb")
def thumb(
    photo_id: uuid.UUID,
    size: Annotated[int, Query()] = 512,
    v: str | None = None,
) -> FileResponse:
    with get_conn() as conn:
        p = _photo(conn, photo_id)
    tp = resolve(p["thumb_path"])
    if tp is None or not tp.exists():
        raise HTTPException(404, "no thumbnail")
    headers = _cache_headers(v is not None)
    if size <= SMALL:
        return FileResponse(_small_thumb(photo_id, tp), media_type="image/webp", headers=headers)
    return FileResponse(tp, media_type="image/jpeg", headers=headers)


@router.get("/photos/{photo_id}/display")
def display(photo_id: uuid.UUID, v: str | None = None) -> FileResponse:
    """A 2048 px JPEG for the lightbox: sharp on a phone screen at a fraction of the
    original's size (and viewable for HEIC / RAW, which browsers mostly can't show)."""
    from workers.media import open_image

    with get_conn() as conn:
        p = _photo(conn, photo_id)
    s = str(photo_id)
    out = get_settings().display_dir / s[:2] / f"{s}_{p['file_hash'][:12].replace(':', '_')}.jpg"
    if not out.exists():
        if p["is_video_frame"]:
            frame = resolve(p["thumb_path"])  # frames are only stored at thumbnail size
            if frame is None:
                raise HTTPException(404, "no frame image")
            src = frame
        else:
            src = Path(p["path"])
            if not src.exists():
                raise HTTPException(404, "file no longer on disk")
        im = open_image(src, max_side=DISPLAY_SIDE)
        im.thumbnail((DISPLAY_SIDE, DISPLAY_SIDE))
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        im.save(tmp, "JPEG", quality=84, progressive=True)
        tmp.replace(out)
    return FileResponse(out, media_type="image/jpeg", headers=_cache_headers(v is not None))


@router.get("/photos/{photo_id}/original", response_model=None)
def original(
    photo_id: uuid.UUID,
    download: bool = False,
    max_side: Annotated[int, Query(ge=256, le=12000)] = 12000,
):
    """The original file (download=1 sends it as an attachment, as-is). For viewing,
    formats browsers can't show (HEIC, RAW) come back as a full-size JPEG.
    Only paths recorded in the database are ever served. Video frames resolve to
    their video, which supports Range requests for seeking."""
    from workers.media import open_image

    with get_conn() as conn:
        p = _photo(conn, photo_id)
        if p["is_video_frame"]:
            p = _photo(conn, p["source_video_id"])
    path = Path(p["path"])
    if not path.exists():
        raise HTTPException(404, "file no longer on disk")
    if download:
        return FileResponse(path, filename=path.name, headers=DAY)
    if p["media_type"] == "video" or p["format"] in _BROWSER_OK:
        return FileResponse(path, headers=DAY)
    im = open_image(path)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg", headers=DAY)


@router.post("/photos/browse")
def browse(
    filters: SearchFilters,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[dict[str, Any]]:
    """Filter-only browsing (timeline view), newest first."""
    where, params = to_sql(filters)
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT p.id, p.path, p.file_hash, p.taken_at, p.place_name, p.camera, p.lens, p.width, p.height, "
            "p.is_video_frame, p.source_video_id, p.frame_ts "
            f"FROM photos p WHERE p.media_type = 'image' {where} "
            "ORDER BY p.taken_at DESC NULLS LAST, p.id LIMIT %(lim)s OFFSET %(off)s",
            {**params, "lim": limit, "off": offset},
        ).fetchall()
    return [{**r, "thumb_url": thumb_url(r["id"], r["file_hash"])} for r in rows]


@router.get("/stats")
def stats() -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM photos WHERE media_type = 'image' AND NOT is_video_frame) AS photos,
              (SELECT count(*) FROM photos WHERE media_type = 'video') AS videos,
              (SELECT count(*) FROM captions) AS captions,
              (SELECT count(*) FROM ocr_text WHERE ran_ocr) AS ocr,
              (SELECT count(*) FROM faces) AS faces,
              (SELECT count(*) FROM people WHERE name IS NOT NULL) AS named_people,
              (SELECT count(*) FROM albums) AS albums,
              (SELECT min(taken_at) FROM photos) AS first_photo,
              (SELECT max(taken_at) FROM photos) AS last_photo
            """
        ).fetchone()
        models = conn.execute(
            "SELECT model, count(*) AS n FROM image_embeddings GROUP BY model ORDER BY model"
        ).fetchall()
    return {**(row or {}), "embeddings": {r["model"]: r["n"] for r in models}}


@router.get("/vocab")
def vocab() -> dict[str, list[str]]:
    """Cameras, lenses, places and people: autocomplete for the filter chips."""
    from api.search.query_parser import load_vocab

    v = load_vocab()
    return {"cameras": v.cameras, "lenses": v.lenses, "places": v.places, "people": v.people}
