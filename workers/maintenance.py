"""Housekeeping for moving to a new machine or recovering from a lost data dir.

relink OLD NEW   the library moved (new drive, new mount point, new OS): rewrite
                 the stored original paths without re-indexing anything
repair           regenerate thumbnails and video-frame stills that are missing
                 (e.g. after restoring the database onto a fresh machine)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from api.db.session import get_conn, one
from api.paths import resolve, stored

log = logging.getLogger(__name__)


def relink(old_prefix: str, new_prefix: str, dry_run: bool = False) -> int:
    old = old_prefix.rstrip("/\\")
    new = new_prefix.rstrip("/\\")
    like = old.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    with get_conn() as conn:
        n = one(
            conn,
            "SELECT count(*) AS n FROM photos WHERE path = %s OR path LIKE %s",
            (old, like + os.sep + "%"),
        )["n"]
        if not dry_run:
            conn.execute(
                "UPDATE photos SET path = %s || substring(path from length(%s) + 1) "
                "WHERE path = %s OR path LIKE %s",
                (new, old, old, like + os.sep + "%"),
            )
            conn.execute(
                "UPDATE imports SET dest_path = %s || substring(dest_path from length(%s) + 1) "
                "WHERE dest_path LIKE %s",
                (new, old, like + os.sep + "%"),
            )
            conn.commit()
    return n


def regeocode() -> int:
    """Recompute place names for every photo with GPS (after a geocoder change)."""
    from workers.geocode import get_geocoder

    geo = get_geocoder()
    if geo is None:
        raise RuntimeError("GeoNames data unavailable")
    with get_conn() as conn:
        rows = conn.execute("SELECT id, lat, lon FROM photos WHERE lat IS NOT NULL").fetchall()
        with conn.cursor() as cur:
            updates = []
            for r in rows:
                pl = geo.lookup(r["lat"], r["lon"])
                updates.append(
                    (
                        pl.name if pl else None,
                        pl.admin1 if pl else None,
                        pl.country if pl else None,
                        r["id"],
                    )
                )
            cur.executemany(
                "UPDATE photos SET place_name = %s, admin1 = %s, country = %s WHERE id = %s",
                updates,
            )
        conn.commit()
    return len(rows)


def repair(progress=None) -> dict[str, int]:
    from workers import media
    from workers.ingest import thumb_path_for

    fixed = missing_original = 0
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, path, thumb_path, media_type, duration_s FROM photos "
            "WHERE NOT is_video_frame AND error IS NULL ORDER BY id"
        ).fetchall()
    from api.config import get_settings

    size = get_settings().thumb_size
    for i, r in enumerate(rows):
        tp = resolve(r["thumb_path"])
        if tp is not None and tp.exists():
            continue
        src = Path(r["path"])
        if not src.exists():
            missing_original += 1
            continue
        try:
            if r["media_type"] == "video":
                from workers.video import frame_at

                im = frame_at(src, min(1.0, (r["duration_s"] or 0) / 2), size)
            else:
                im = media.make_thumbnail(media.open_image(src, max_side=size * 2), size)
            if im is None:
                continue
            out = thumb_path_for(r["id"])
            out.parent.mkdir(parents=True, exist_ok=True)
            im.save(out, "JPEG", quality=85)
            with get_conn() as conn:
                conn.execute(
                    "UPDATE photos SET thumb_path = %s WHERE id = %s", (stored(out), r["id"])
                )
                conn.commit()
            fixed += 1
        except Exception as e:
            log.warning("could not rebuild thumbnail for %s: %s", src, e)
        if progress:
            progress("repair", i + 1, len(rows))
    # video frame stills can't be regenerated one by one: re-extract those videos
    with get_conn() as conn:
        frames = conn.execute(
            "SELECT DISTINCT source_video_id AS vid, thumb_path FROM photos WHERE is_video_frame"
        ).fetchall()
        stale = {
            f["vid"]
            for f in frames
            if not (resolve(f["thumb_path"]) or Path("/nonexistent")).exists()
        }
        if stale:
            conn.execute("DELETE FROM photos WHERE source_video_id = ANY(%s)", (list(stale),))
            conn.execute(
                "UPDATE photos SET frames_indexed_at = NULL WHERE id = ANY(%s)", (list(stale),)
            )
            conn.commit()
    return {
        "thumbnails_rebuilt": fixed,
        "originals_missing": missing_original,
        "videos_to_reextract": len(stale),
    }
