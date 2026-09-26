"""Library scanner: walk folders, detect new / changed / moved / deleted files,
extract metadata and thumbnails, and upsert `photos` rows.

Re-scans are incremental. A file is only re-read when its (size, mtime)
changed, only re-processed when its content hash changed, and a file that
moved keeps its id (and therefore its embeddings, captions and faces).
Downstream stages (embed, caption, OCR, faces, video frames) pick up
whatever is missing on their own, so the scanner never needs to know about them.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

from api.config import get_settings
from api.db.session import Conn
from api.paths import stored
from workers import media
from workers.geocode import get_geocoder

log = logging.getLogger(__name__)

COMMIT_EVERY = 200


@dataclass
class ScanStats:
    seen: int = 0
    new: int = 0
    changed: int = 0
    touched: int = 0
    moved: int = 0
    unchanged: int = 0
    deleted: int = 0
    errors: int = 0
    photo_ids: list[uuid.UUID] = field(default_factory=list)  # new or changed

    def summary(self) -> dict[str, int]:
        return {k: v for k, v in self.__dict__.items() if k != "photo_ids"}


def iter_media(roots: Iterable[Path]) -> Iterator[Path]:
    for root in roots:
        root = Path(root).expanduser().resolve()
        if root.is_file():
            if media.media_kind(root):
                yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                p = Path(dirpath) / fn
                if media.media_kind(p):
                    yield p


def thumb_path_for(photo_id: uuid.UUID) -> Path:
    s = str(photo_id)
    return get_settings().thumbs_dir / s[:2] / f"{s}.jpg"


def extract(path: Path, photo_id: uuid.UUID, file_hash: str) -> dict[str, Any]:
    """Everything we store about one file. Runs in a worker thread."""
    s = get_settings()
    st = path.stat()
    kind = media.media_kind(path)
    row: dict[str, Any] = {
        "id": photo_id,
        "path": str(path),
        "file_hash": file_hash,
        "size_bytes": st.st_size,
        "mtime": st.st_mtime,
        "media_type": "video" if kind == "video" else "image",
        "format": media.file_format(path),
        "error": None,
    }
    thumb = None
    if kind == "video":
        from workers import video

        info = video.probe(path)
        created = info.created_at
        row.update(
            taken_at=created,
            tz_offset_min=None,
            lat=info.lat,
            lon=info.lon,
            width=info.width,
            height=info.height,
            duration_s=info.duration_s,
            keywords=media.read_keywords(path),
        )
        thumb = video.frame_at(path, min(1.0, (info.duration_s or 0) / 2), s.thumb_size)
    else:
        ex = media.read_exif(path)
        row.update(
            {
                k: getattr(ex, k)
                for k in (
                    "taken_at",
                    "tz_offset_min",
                    "lat",
                    "lon",
                    "camera",
                    "lens",
                    "focal_length",
                    "focal_length_35mm",
                    "aperture",
                    "iso",
                    "shutter",
                    "exposure_s",
                    "orientation",
                    "keywords",
                )
            }
        )
        im = media.open_image(path, max_side=s.thumb_size * 2)
        if ex.width and ex.height:
            w, h = ex.width, ex.height
            if (ex.orientation or 1) in (5, 6, 7, 8):
                w, h = h, w
        elif kind == "raw":
            w, h = im.size
        else:
            from PIL import Image, ImageOps

            with Image.open(path) as probe:
                w, h = ImageOps.exif_transpose(probe).size if ex.orientation else probe.size
        row["width"], row["height"] = w, h
        thumb = media.make_thumbnail(im, s.thumb_size)
        if row["taken_at"] is None:
            # no EXIF date (screenshots, scans, exports): the file's mtime is the best guess
            from datetime import datetime

            row["taken_at"] = datetime.fromtimestamp(st.st_mtime, tz=UTC)

    if thumb is not None:
        tp = thumb_path_for(photo_id)
        tp.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(tp, "JPEG", quality=85)
        row["thumb_path"] = stored(tp)
        row["phash"] = media.perceptual_hash(thumb)
        row["sharpness"] = media.sharpness(thumb)

    row.update(place_name=None, admin1=None, country=None)
    if row.get("lat") is not None and row.get("lon") is not None:
        geo = get_geocoder()
        if geo is not None:
            place = geo.lookup(row["lat"], row["lon"])
            if place:
                row.update(place_name=place.name, admin1=place.admin1, country=place.country)
    return row


PHOTO_COLS = [
    "id", "path", "file_hash", "size_bytes", "mtime", "media_type", "format", "taken_at",
    "tz_offset_min", "lat", "lon", "place_name", "admin1", "country", "camera", "lens",
    "focal_length", "focal_length_35mm", "aperture", "iso", "shutter", "exposure_s", "width",
    "height", "orientation", "keywords", "duration_s", "thumb_path", "phash", "sharpness", "error",
]  # fmt: skip


def upsert_photo(conn: Conn, row: dict[str, Any]) -> None:
    data = {c: row.get(c) for c in PHOTO_COLS}
    data["keywords"] = data["keywords"] or []
    cols = ", ".join(PHOTO_COLS)
    vals = ", ".join(f"%({c})s" for c in PHOTO_COLS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in PHOTO_COLS if c not in ("id", "path"))
    conn.execute(
        f"INSERT INTO photos ({cols}) VALUES ({vals}) "
        f"ON CONFLICT (path) DO UPDATE SET {updates}, indexed_at = now()",
        data,
    )


def clear_derived(conn: Conn, photo_id: uuid.UUID) -> None:
    """Content changed: drop everything computed from the old pixels."""
    for table in ("image_embeddings", "captions", "ocr_text", "faces", "photo_groups"):
        conn.execute(f"DELETE FROM {table} WHERE photo_id = %s", (photo_id,))
    conn.execute("DELETE FROM photos WHERE source_video_id = %s", (photo_id,))
    conn.execute(
        "UPDATE photos SET faces_indexed_at = NULL, frames_indexed_at = NULL WHERE id = %s",
        (photo_id,),
    )


def _under(path: str, roots: list[Path]) -> bool:
    return any(path == str(r) or path.startswith(str(r) + os.sep) for r in roots)


def scan(
    roots: Iterable[Path] | None = None,
    conn: Conn | None = None,
    *,
    workers: int = 8,
    remove_missing: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> ScanStats:
    from api.db.session import get_conn

    s = get_settings()
    roots = [Path(r).expanduser().resolve() for r in (roots or s.photo_roots)]
    if not roots:
        raise ValueError("no photo roots given (set PS_PHOTO_ROOTS or pass paths)")
    if conn is None:
        with get_conn() as c:
            return scan(roots, c, workers=workers, remove_missing=remove_missing, progress=progress)

    stats = ScanStats()
    existing = {
        r["path"]: r
        for r in conn.execute(
            "SELECT id, path, file_hash, size_bytes, mtime FROM photos WHERE NOT is_video_frame"
        ).fetchall()
        if _under(r["path"], roots)
    }

    # 1. stat everything, keep only files whose size/mtime moved
    candidates: list[tuple[Path, os.stat_result]] = []
    seen_paths: set[str] = set()
    for p in iter_media(roots):
        stats.seen += 1
        sp = str(p)
        seen_paths.add(sp)
        try:
            st = p.stat()
        except OSError:
            stats.errors += 1
            continue
        old = existing.get(sp)
        if (
            old
            and old["size_bytes"] == st.st_size
            and abs((old["mtime"] or 0) - st.st_mtime) < 1e-3
        ):
            stats.unchanged += 1
            continue
        candidates.append((p, st))

    missing = {p: r for p, r in existing.items() if p not in seen_paths and not Path(p).exists()}

    # 2. hash candidates
    def _hash(item: tuple[Path, os.stat_result]) -> tuple[Path, os.stat_result, str | None]:
        p, st = item
        try:
            return p, st, media.hash_file(p, st.st_size)
        except OSError:
            return p, st, None

    with ThreadPoolExecutor(workers) as ex:
        hashed = list(ex.map(_hash, candidates))

    # 3. classify: touched / moved / changed / new
    missing_by_hash: dict[str, dict] = {}
    for r in missing.values():
        missing_by_hash.setdefault(r["file_hash"], r)
    todo: list[tuple[Path, uuid.UUID, str, bool]] = []  # path, id, hash, content_changed
    for p, st, h in hashed:
        if h is None:
            stats.errors += 1
            continue
        old = existing.get(str(p))
        if old is not None:
            if old["file_hash"] == h:
                conn.execute(
                    "UPDATE photos SET mtime = %s, size_bytes = %s WHERE id = %s",
                    (st.st_mtime, st.st_size, old["id"]),
                )
                stats.touched += 1
            else:
                todo.append((p, old["id"], h, True))
        elif h in missing_by_hash:
            moved = missing_by_hash.pop(h)
            missing.pop(moved["path"], None)
            conn.execute(
                "UPDATE photos SET path = %s, mtime = %s WHERE id = %s",
                (str(p), st.st_mtime, moved["id"]),
            )
            # video frames encode the parent path in theirs
            conn.execute(
                "UPDATE photos SET path = %s || substring(path from length(%s) + 1) "
                "WHERE source_video_id = %s",
                (str(p), moved["path"], moved["id"]),
            )
            stats.moved += 1
        else:
            todo.append((p, uuid.uuid4(), h, False))
    conn.commit()

    # 4. extract + upsert
    def _extract(item: tuple[Path, uuid.UUID, str, bool]) -> tuple[dict | None, str | None]:
        p, pid, h, _ = item
        try:
            return extract(p, pid, h), None
        except Exception as e:  # corrupt file, unsupported codec, ...
            log.warning("failed to read %s: %s", p, e)
            return None, f"{type(e).__name__}: {e}"

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for (p, pid, h, changed), (row, err) in zip(todo, ex.map(_extract, todo), strict=True):
            if changed:
                clear_derived(conn, pid)
            if row is None:
                stats.errors += 1
                row = {
                    "id": pid, "path": str(p), "file_hash": h, "error": err,
                    "media_type": "image", "format": media.file_format(p),
                    "size_bytes": p.stat().st_size, "mtime": p.stat().st_mtime,
                }  # fmt: skip
            upsert_photo(conn, row)
            if row.get("error") is None:
                stats.photo_ids.append(pid)
            if changed:
                stats.changed += 1
            else:
                stats.new += 1
            done += 1
            if done % COMMIT_EVERY == 0:
                conn.commit()
            if progress:
                progress("extract", done, len(todo))
    conn.commit()

    # 5. deletions
    if remove_missing and missing:
        ids = [r["id"] for r in missing.values()]
        conn.execute("DELETE FROM photos WHERE id = ANY(%s)", (ids,))
        for pid in ids:
            thumb_path_for(pid).unlink(missing_ok=True)
        stats.deleted = len(ids)
        conn.commit()
    return stats


def remove_paths(paths: Iterable[str]) -> int:
    """Delete rows for files that no longer exist (used by the watcher)."""
    from api.db.session import get_conn

    paths = [p for p in paths if not Path(p).exists()]
    if not paths:
        return 0
    with get_conn() as conn:
        rows = conn.execute(
            "DELETE FROM photos WHERE path = ANY(%s) RETURNING id", (list(paths),)
        ).fetchall()
        conn.commit()
    for r in rows:
        thumb_path_for(r["id"]).unlink(missing_ok=True)
    return len(rows)
