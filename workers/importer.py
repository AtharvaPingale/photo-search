"""Auto import from a phone's camera-roll backup folder.

Point PS_IMPORT_DIRS at wherever your phone backs up to (PhotoSync, Syncthing,
Resilio, a synced iCloud Photos / Google Photos folder, an SD card dump...).
New files are copied (or moved) into the library as

    <import_dest>/<YYYY>/<YYYY-MM>/<original name>

and then indexed like any other photo.

Details that matter for a backup folder:
  * dedupe by content hash: backup apps re-upload, rename (IMG_0001 (1).HEIC)
    and resync; the same bytes are imported once.
  * a ledger (imports table) remembers every hash ever seen, so deleting a photo
    from the library doesn't make it come back on the next sync.
  * files still being written are skipped until their size is stable, and
    temp/partial names are ignored.
  * sidecars (.xmp, .aae) travel with their photo.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from api.config import get_settings
from api.db.session import get_conn
from workers import media

log = logging.getLogger(__name__)

PARTIAL_SUFFIXES = {".tmp", ".part", ".partial", ".crdownload", ".download", ".syncthing"}
SIDECARS = (".xmp", ".XMP", ".aae", ".AAE")
STABLE_S = 3.0


@dataclass
class ImportStats:
    imported: list[Path] = field(default_factory=list)
    duplicates: int = 0
    already_seen: int = 0
    not_ready: int = 0
    errors: int = 0

    def summary(self) -> dict[str, int]:
        return {
            "imported": len(self.imported), "duplicates": self.duplicates,
            "already_seen": self.already_seen, "not_ready": self.not_ready, "errors": self.errors,
        }  # fmt: skip


def destination_root() -> Path:
    s = get_settings()
    dest = s.import_dest or (s.photo_roots[0] if s.photo_roots else None)
    if dest is None:
        raise ValueError("set PS_IMPORT_DEST or PS_PHOTO_ROOTS so imports have somewhere to go")
    return Path(dest).expanduser().resolve()


def is_candidate(p: Path) -> bool:
    name = p.name
    if name.startswith((".", "~")) or p.suffix.lower() in PARTIAL_SUFFIXES:
        return False
    return media.media_kind(p) is not None


def is_stable(p: Path, wait_s: float = STABLE_S) -> bool:
    """Not modified for `wait_s` seconds. A file still being uploaded keeps getting
    written; it is simply left for the next sweep (the watcher runs one after every
    burst of activity, plus a periodic one)."""
    try:
        return time.time() - p.stat().st_mtime >= wait_s
    except OSError:
        return False


def taken_date(p: Path) -> datetime:
    if media.media_kind(p) == "video":
        from workers.video import probe

        created = probe(p).created_at
        if created:
            return created
    else:
        ex = media.read_exif(p)
        if ex.taken_at:
            return ex.taken_at
    return datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)


def target_path(root: Path, src: Path, when: datetime, file_hash: str) -> Path:
    """<root>/YYYY/YYYY-MM/name, adding -1, -2... if a different file has that name."""
    folder = root / f"{when:%Y}" / f"{when:%Y-%m}"
    cand = folder / src.name
    n = 1
    while cand.exists():
        if media.hash_file(cand) == file_hash:
            return cand  # identical file already there (e.g. an interrupted earlier import)
        cand = folder / f"{src.stem}-{n}{src.suffix}"
        n += 1
    return cand


def import_new(
    dirs: list[Path] | None = None, *, mode: str | None = None, wait_s: float = STABLE_S
) -> ImportStats:
    s = get_settings()
    dirs = [Path(d).expanduser().resolve() for d in (dirs or s.import_dirs)]
    mode = mode or s.import_mode
    root = destination_root()
    stats = ImportStats()
    with get_conn() as conn:
        # failed imports are retried on the next run; everything else is never imported twice
        seen = {
            r["file_hash"]
            for r in conn.execute("SELECT file_hash FROM imports WHERE status <> 'error'")
        }
        in_library = {r["file_hash"] for r in conn.execute("SELECT file_hash FROM photos")}
    for d in dirs:
        if not d.exists():
            log.warning("import folder %s does not exist", d)
            continue
        # shortest name first, so "IMG_1.JPG" is kept over a backup app's "IMG_1 (1).JPG"
        files = sorted((p for p in d.rglob("*") if p.is_file() and is_candidate(p)),
                       key=lambda p: (str(p.parent), len(p.stem), p.name))  # fmt: skip
        for src in files:
            if root in src.parents:
                continue  # never import from inside the library itself
            if not is_stable(src, wait_s):
                stats.not_ready += 1
                continue
            try:
                h = media.hash_file(src)
            except OSError:
                stats.errors += 1
                continue
            if h in seen:
                stats.already_seen += 1
                continue
            status, dest, err = "imported", None, None
            if h in in_library:
                status = "duplicate"
                stats.duplicates += 1
            else:
                try:
                    dest = target_path(root, src, taken_date(src), h)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    transfer = shutil.move if mode == "move" else shutil.copy2
                    if not dest.exists():
                        transfer(src, dest)
                    for ext in SIDECARS:  # IMG_1.xmp / IMG_1.CR2.xmp / IMG_1.AAE travel along
                        for side, target in ((src.with_suffix(ext), dest.with_suffix(ext)),
                                             (Path(f"{src}{ext}"), Path(f"{dest}{ext}"))):  # fmt: skip
                            if side.exists() and not target.exists():
                                transfer(side, target)
                    stats.imported.append(dest)
                except Exception as e:
                    status, err = "error", f"{type(e).__name__}: {e}"
                    stats.errors += 1
                    log.warning("import failed for %s: %s", src, e)
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO imports (file_hash, source_path, dest_path, status, error) VALUES (%s,%s,%s,%s,%s) "
                    "ON CONFLICT (file_hash) DO UPDATE SET source_path = EXCLUDED.source_path, "
                    "dest_path = EXCLUDED.dest_path, status = EXCLUDED.status, error = EXCLUDED.error, "
                    "imported_at = now()",
                    (h, str(src), str(dest) if dest else None, status, err),
                )
                conn.commit()
            seen.add(h)
            if status == "imported":
                in_library.add(h)
    return stats


def import_and_index(
    dirs: list[Path] | None = None, *, inline: bool = True, index: bool = True
) -> dict:
    """Import, then index just the imported files (inline, or via the GPU queue).
    index=False when a watcher on the library will pick the new files up anyway."""
    from workers.ingest import scan

    stats = import_new(dirs)
    out: dict = {"import": stats.summary()}
    if stats.imported and index:
        scanned = scan(stats.imported, remove_missing=False)
        out["scan"] = scanned.summary()
        if scanned.photo_ids:
            if inline:
                from workers.pipeline import run_pipeline

                out["pipeline"] = run_pipeline(
                    scan=False, ids=scanned.photo_ids, faces=bool(get_settings().face_roots)
                )
            else:
                from workers.queue import enqueue_ids

                enqueue_ids(scanned.photo_ids)
    return out
