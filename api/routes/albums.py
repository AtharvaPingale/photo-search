from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from api.db.session import get_conn
from api.urls import thumb_url

router = APIRouter(tags=["albums"])


@router.get("/albums")
def list_albums() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT a.id, a.title, a.summary, a.start_at, a.end_at, a.place_name, a.cover_photo_id, a.auto, "
            "count(ap.photo_id) AS n_photos FROM albums a LEFT JOIN album_photos ap ON ap.album_id = a.id "
            "GROUP BY a.id ORDER BY a.start_at DESC NULLS LAST"
        ).fetchall()
    return [
        {
            **r,
            "cover_url": thumb_url(r["cover_photo_id"], size=512) if r["cover_photo_id"] else None,
        }
        for r in rows
    ]


@router.get("/albums/{album_id}")
def album(album_id: uuid.UUID) -> dict[str, Any]:
    with get_conn() as conn:
        a = conn.execute("SELECT * FROM albums WHERE id = %s", (album_id,)).fetchone()
        if a is None:
            raise HTTPException(404, "no such album")
        photos = conn.execute(
            "SELECT p.id, p.path, p.file_hash, p.taken_at, p.place_name, p.width, p.height, p.is_video_frame "
            "FROM album_photos ap "
            "JOIN photos p ON p.id = ap.photo_id WHERE ap.album_id = %s ORDER BY p.taken_at, p.id",
            (album_id,),
        ).fetchall()
    return {**a, "photos": [{**p, "thumb_url": thumb_url(p["id"], p["file_hash"])} for p in photos]}
