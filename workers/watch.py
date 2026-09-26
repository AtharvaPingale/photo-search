"""Incremental indexing with watchdog: new photos are searchable within a minute.

File events are debounced (cameras and sync tools write in bursts, and a file
may be written in several chunks), then handled as one batch:
created/modified -> scan just those files; moved -> update the path in place
(keeps embeddings); deleted -> drop the rows. The new ids go straight to the
GPU queue, or run inline with --inline when Redis isn't running.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from api.config import get_settings
from workers.media import media_kind

log = logging.getLogger(__name__)
DEBOUNCE_S = 5.0
IMPORT_SWEEP_S = 600.0


class _Handler(FileSystemEventHandler):
    def __init__(self, sink: Batcher):
        self.sink = sink

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = str(event.src_path)
        if event.event_type in ("created", "modified", "closed"):
            if media_kind(Path(src)):
                self.sink.add("upsert", src)
        elif event.event_type == "deleted":
            if media_kind(Path(src)):
                self.sink.add("delete", src)
        elif event.event_type == "moved":
            dest = str(event.dest_path)
            if media_kind(Path(dest)) or media_kind(Path(src)):
                self.sink.add("move", (src, dest))


class _ImportHandler(FileSystemEventHandler):
    """Any file activity in a camera-roll backup folder schedules an import sweep."""

    def __init__(self, sink: Batcher):
        self.sink = sink

    def on_any_event(self, event: FileSystemEvent) -> None:
        if not event.is_directory and event.event_type in (
            "created",
            "modified",
            "closed",
            "moved",
        ):
            self.sink.add("import", None)


class Batcher:
    def __init__(self, inline: bool, index_imports: bool = True):
        self.inline = inline
        self.index_imports = index_imports
        self.import_due = False
        self.last_import = 0.0
        self.upserts: set[str] = set()
        self.deletes: set[str] = set()
        self.moves: list[tuple[str, str]] = []
        self.last = 0.0
        self.lock = threading.Lock()

    def add(self, kind: str, item) -> None:
        with self.lock:
            if kind == "upsert":
                self.upserts.add(item)
                self.deletes.discard(item)
            elif kind == "delete":
                self.deletes.add(item)
                self.upserts.discard(item)
            elif kind == "import":
                self.import_due = True
            else:
                self.moves.append(item)
            self.last = time.monotonic()

    def flush_if_quiet(self) -> None:
        self._maybe_import()
        with self.lock:
            if (
                not (self.upserts or self.deletes or self.moves)
                or time.monotonic() - self.last < DEBOUNCE_S
            ):
                return
            ups, dels, moves = self.upserts, self.deletes, self.moves
            self.upserts, self.deletes, self.moves = set(), set(), []
        try:
            process(ups, dels, moves, inline=self.inline)
        except Exception:
            log.exception("watch batch failed")

    def _maybe_import(self) -> None:
        with self.lock:
            quiet = time.monotonic() - self.last >= DEBOUNCE_S
            # sweep on activity, and every IMPORT_SWEEP_S regardless (missed events, network drives)
            due = (
                self.import_due and quiet
            ) or time.monotonic() - self.last_import > IMPORT_SWEEP_S
            if not due or not get_settings().import_dirs:
                return
            self.import_due = False
            self.last_import = time.monotonic()
        try:
            from workers.importer import import_and_index

            out = import_and_index(inline=self.inline, index=self.index_imports)
            if out["import"]["imported"]:
                log.warning("imported: %s", out["import"])
        except Exception:
            log.exception("import sweep failed")


def process(
    upserts: set[str], deletes: set[str], moves: list[tuple[str, str]], *, inline: bool
) -> dict:
    from api.db.session import get_conn
    from workers.ingest import remove_paths, scan

    moved = 0
    with get_conn() as conn:
        for src, dest in moves:
            n = conn.execute("UPDATE photos SET path = %s WHERE path = %s", (dest, src)).rowcount
            if n:
                conn.execute(
                    "UPDATE photos SET path = %s || substring(path from length(%s) + 1) "
                    "WHERE path LIKE %s AND is_video_frame",
                    (dest, src, src.replace("%", "\\%") + "#t=%"),
                )
                moved += n
            elif media_kind(Path(dest)):
                upserts.add(dest)  # moved in from outside the library
        conn.commit()
    existing = [Path(p) for p in upserts if Path(p).exists()]
    stats = scan(existing, remove_missing=False) if existing else None
    removed = remove_paths(deletes)
    ids = stats.photo_ids if stats else []
    if ids:
        if inline:
            from workers.pipeline import run_pipeline

            run_pipeline(scan=False, ids=ids, faces=bool(get_settings().face_roots))
        else:
            from workers.queue import enqueue_ids

            enqueue_ids(ids)
    out = {"indexed": len(ids), "moved": moved, "removed": removed}
    log.info("watch batch: %s", out)
    return out


def watch(roots: Iterable[Path] | None = None, *, inline: bool = False, poll: bool = False) -> None:
    """poll=True uses stat polling instead of inotify/FSEvents: needed for network
    shares and for Windows drives under WSL (/mnt/c), which never deliver events."""
    s = get_settings()
    roots = [Path(r).expanduser().resolve() for r in (roots or s.photo_roots)]
    if not roots:
        raise SystemExit("no folders to watch (set PS_PHOTO_ROOTS or pass paths)")
    imports = [
        Path(d).expanduser().resolve() for d in s.import_dirs if Path(d).expanduser().exists()
    ]
    index_imports = True
    if imports:
        from workers.importer import destination_root

        # if imports land inside a watched folder, the library watcher indexes them
        dest = destination_root()
        index_imports = not any(dest == r or r in dest.parents for r in roots)
    batcher = Batcher(inline, index_imports=index_imports)
    obs: BaseObserver
    if poll:
        from watchdog.observers.polling import PollingObserver

        obs = PollingObserver(timeout=15)
    else:
        obs = Observer()
    for r in roots:
        obs.schedule(_Handler(batcher), str(r), recursive=True)
    for d in imports:
        obs.schedule(_ImportHandler(batcher), str(d), recursive=True)
    obs.start()
    log.warning(
        "watching %s%s (%s)", ", ".join(map(str, roots)),
        f"; importing from {', '.join(map(str, imports))}" if imports else "",
        "inline" if inline else "queue",
    )  # fmt: skip
    try:
        while True:
            time.sleep(1)
            batcher.flush_if_quiet()
    except KeyboardInterrupt:
        pass
    finally:
        obs.stop()
        obs.join()
