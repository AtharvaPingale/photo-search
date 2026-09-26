"""`photo-search` command line: indexing stages, search, maintenance, serving."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
db_app = typer.Typer(no_args_is_help=True, help="Database management")
faces_app = typer.Typer(no_args_is_help=True, help="Face detection, clustering and privacy")
app.add_typer(db_app, name="db")
app.add_typer(faces_app, name="faces")
console = Console()


def _progress():
    p = Progress(
        TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console, transient=True,
    )  # fmt: skip
    tasks: dict[str, TaskID] = {}

    def cb(stage: str, done: int, total: int) -> None:
        if stage not in tasks:
            tasks[stage] = p.add_task(stage, total=total)
        p.update(tasks[stage], completed=done, total=total)

    return p, cb


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


# ------------------------------------------------------------------ db


@db_app.command("migrate")
def db_migrate() -> None:
    """Apply pending schema migrations."""
    from api.config import get_settings
    from api.db.migrate import migrate

    applied = migrate(get_settings().database_url)
    console.print(f"applied: {applied or 'nothing (up to date)'}")


@db_app.command("backup")
def db_backup(prune_old: Annotated[bool, typer.Option("--prune/--no-prune")] = True) -> None:
    """pg_dump the database into PS_BACKUPS_DIR (and prune old dumps)."""
    from api.db.backup import backup, prune

    out = backup()
    console.print(f"backup: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if prune_old:
        for p in prune():
            console.print(f"pruned {p.name}")


@db_app.command("backups")
def db_backups() -> None:
    """List backups, newest first."""
    from datetime import datetime

    from api.db.backup import list_backups

    t = Table("file", "MB", "taken")
    for p, mb, mtime in list_backups():
        t.add_row(p.name, f"{mb:.1f}", f"{datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M}")
    console.print(t)


@db_app.command("restore")
def db_restore(
    dump: Annotated[Path | None, typer.Argument(help="default: the newest backup")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Replace the database with a backup (stop the API and workers first)."""
    from api.db.backup import list_backups, restore

    if dump is None:
        backups = list_backups()
        if not backups:
            console.print("no backups found")
            raise typer.Exit(1)
        dump = backups[0][0]
    if not yes:
        typer.confirm(f"Replace the current database with {dump.name}?", abort=True)
    restore(dump)
    console.print(f"restored {dump.name}; run `photo-search repair` if thumbnails are missing")


@db_app.command("stats")
def db_stats() -> None:
    """Row counts and index coverage per stage."""
    from api.db.session import get_conn

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM photos WHERE media_type='image' AND NOT is_video_frame) AS photos,
              (SELECT count(*) FROM photos WHERE media_type='video') AS videos,
              (SELECT count(*) FROM photos WHERE is_video_frame) AS video_frames,
              (SELECT count(*) FROM photos WHERE error IS NOT NULL) AS errors,
              (SELECT count(*) FROM photos WHERE lat IS NOT NULL) AS with_gps,
              (SELECT count(*) FROM captions) AS captions,
              (SELECT count(*) FROM ocr_text WHERE ran_ocr) AS ocr_run,
              (SELECT count(*) FROM faces) AS faces,
              (SELECT count(DISTINCT cluster_id) FROM faces WHERE cluster_id IS NOT NULL) AS clusters,
              (SELECT count(*) FROM albums) AS albums,
              pg_size_pretty(pg_database_size(current_database())) AS db_size
            """
        ).fetchone()
        per_model = conn.execute(
            "SELECT model, count(*) n FROM image_embeddings GROUP BY model ORDER BY model"
        ).fetchall()
    t = Table("metric", "value")
    for k, v in (rows or {}).items():
        t.add_row(k, str(v))
    for r in per_model:
        t.add_row(f"embeddings[{r['model']}]", str(r["n"]))
    console.print(t)


# ------------------------------------------------------------------ indexing


@app.command()
def scan(
    paths: Annotated[
        list[Path] | None, typer.Argument(help="folders (default: PS_PHOTO_ROOTS)")
    ] = None,
    no_delete: Annotated[bool, typer.Option(help="keep rows for files that disappeared")] = False,
    workers: int = 8,
) -> None:
    """Scan folders: metadata, GPS -> place, thumbnails. Incremental."""
    from workers.ingest import scan as do_scan

    p, cb = _progress()
    with p:
        stats = do_scan(paths or None, workers=workers, remove_missing=not no_delete, progress=cb)
    console.print(stats.summary())


@app.command()
def embed(
    model: Annotated[
        str | None, typer.Option(help="image model name (default PS_IMAGE_MODEL)")
    ] = None,
    limit: int | None = None,
    batch_size: int | None = None,
) -> None:
    """CLIP-embed every photo missing an embedding for MODEL."""
    from workers.embed import embed_pending

    p, cb = _progress()
    with p:
        n = embed_pending(model, limit=limit, batch_size=batch_size, progress=cb)
    console.print(f"embedded {n} photos")


@app.command()
def caption(limit: int | None = None, batch_size: int | None = None) -> None:
    """Caption photos with the local VLM and embed the captions."""
    from workers.caption import caption_pending

    p, cb = _progress()
    with p:
        n = caption_pending(limit=limit, batch_size=batch_size, progress=cb)
    console.print(f"captioned {n} photos")


@app.command()
def ocr(limit: int | None = None, threshold: float | None = None) -> None:
    """OCR photos the text gate flags as likely to contain text."""
    from workers.ocr import ocr_pending

    p, cb = _progress()
    with p:
        stats = ocr_pending(limit=limit, threshold=threshold, progress=cb)
    console.print(stats)


@app.command()
def videos(limit: int | None = None) -> None:
    """Extract scene frames from videos (then run `embed` etc. on them)."""
    from workers.video import index_videos

    console.print(f"extracted {index_videos(limit)} frames")


@app.command()
def index(
    paths: Annotated[list[Path] | None, typer.Argument()] = None,
    captions: bool = True,
    ocr_: Annotated[bool, typer.Option("--ocr/--no-ocr")] = True,
    faces: bool = False,
    queue: Annotated[bool, typer.Option(help="enqueue on Redis instead of running inline")] = False,
) -> None:
    """Run the whole pipeline: scan -> video frames -> embed -> caption -> OCR -> faces."""
    if queue:
        from workers.queue import enqueue_pipeline

        console.print(enqueue_pipeline(paths or None, captions=captions, ocr=ocr_, faces=faces))
        return
    from workers.pipeline import run_pipeline

    p, cb = _progress()
    with p:
        out = run_pipeline(paths or None, captions=captions, ocr=ocr_, faces=faces, progress=cb)
    console.print(out)


# ------------------------------------------------------------------ faces


@faces_app.command("detect")
def faces_detect(limit: int | None = None) -> None:
    """Detect + embed faces (only under PS_FACE_ROOTS: faces are opt-in per folder)."""
    from workers.faces import detect_pending

    p, cb = _progress()
    with p:
        n = detect_pending(limit=limit, progress=cb)
    console.print(f"processed {n} photos")


@faces_app.command("cluster")
def faces_cluster(min_cluster_size: int = 4) -> None:
    """(Re)cluster face embeddings with HDBSCAN, keeping names and manual edits."""
    from workers.faces import cluster_faces

    console.print(cluster_faces(min_cluster_size=min_cluster_size))


@faces_app.command("wipe")
def faces_wipe(yes: Annotated[bool, typer.Option("--yes", "-y")] = False) -> None:
    """Delete ALL face data: detections, embeddings, clusters, names, crops."""
    from workers.faces import wipe_faces

    if not yes:
        typer.confirm("Delete all face data (detections, clusters, names, crops)?", abort=True)
    console.print(wipe_faces())


@faces_app.command("name")
def faces_name(
    cluster_id: int, name: str, alias: Annotated[list[str] | None, typer.Option()] = None
) -> None:
    """Name a face cluster, e.g. `faces name 3 Rohan` or `faces name 0 Atharva --alias me`."""
    from workers.faces import name_cluster

    name_cluster(cluster_id, name, alias or [])
    console.print("ok")


# ------------------------------------------------------------------ organise


@app.command()
def organize(
    duplicates: bool = True, bursts: bool = True, albums: bool = True, llm_titles: bool = True
) -> None:
    """Near-duplicates, bursts (with best pick) and auto albums."""
    from workers import organize as org

    if duplicates:
        console.print({"duplicates": org.find_duplicates()})
    if bursts:
        console.print({"bursts": org.find_bursts()})
    if albums:
        console.print({"albums": org.build_albums(llm_titles=llm_titles)})


# ------------------------------------------------------------------ search / agent


@app.command()
def search(
    query: str,
    k: int = 10,
    parse: str = "auto",
    model: str | None = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Search from the terminal."""
    from api.search.engine import SearchRequest
    from api.search.engine import search as do_search

    resp = do_search(SearchRequest(q=query, k=k, parse=parse, model=model))
    if as_json:
        print(resp.model_dump_json(indent=2))
        return
    console.print(
        f"[bold]semantic:[/] {resp.parsed.semantic!r}  [bold]filters:[/] {resp.parsed.filters.chips()}  "
        f"[bold]via:[/] {resp.parsed.source}  [bold]signals:[/] {resp.signals}"
        + ("  [yellow](fallback: filters matched nothing)[/]" if resp.fallback_used else "")
    )
    t = Table("#", "score", "taken", "place", "camera / lens", "path")
    for i, h in enumerate(resp.hits, 1):
        t.add_row(
            str(i), f"{h.score:.4f}", str(h.taken_at)[:16], h.place_name or "",
            " / ".join(x for x in (h.camera, h.lens) if x),
            h.path + (f" @{h.frame_ts:.1f}s" if h.frame_ts is not None else ""),
        )  # fmt: skip
    console.print(t)
    console.print(f"timings (ms): {resp.timings_ms}")


@app.command()
def ask(question: str, as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Ask the agent a question about your library."""
    from api.agent.graph import run_agent

    ans = run_agent(question)
    if as_json:
        print(json.dumps(ans.model_dump(mode="json"), indent=2))
        return
    console.print(ans.answer)
    for ev in ans.evidence:
        console.print(f"  [dim]{ev.photo_id}[/]  {ev.note}")
    console.print(f"[dim]{ans.tool_calls} tool calls, grounded={ans.grounded}[/]")


@app.command()
def relink(old_prefix: str, new_prefix: str, dry_run: bool = False) -> None:
    """The library moved: rewrite stored paths OLD_PREFIX/... -> NEW_PREFIX/... (no re-indexing)."""
    from workers.maintenance import relink as do_relink

    n = do_relink(old_prefix, new_prefix, dry_run=dry_run)
    console.print(f"{'would relink' if dry_run else 'relinked'} {n} files")


@app.command()
def regeocode() -> None:
    """Recompute place names from GPS for the whole library."""
    from workers.maintenance import regeocode as do_regeocode

    console.print(f"updated {do_regeocode()} photos")


@app.command()
def repair() -> None:
    """Rebuild missing thumbnails (e.g. after restoring onto a new machine)."""
    from workers.maintenance import repair as do_repair

    p, cb = _progress()
    with p:
        out = do_repair(progress=cb)
    console.print(out)
    if out["videos_to_reextract"]:
        console.print("then run: photo-search videos && photo-search embed")


# ------------------------------------------------------------------ serving


@app.command()
def token() -> None:
    """Generate a random API token for PS_AUTH_TOKEN."""
    import secrets

    console.print(secrets.token_urlsafe(32))


@app.command()
def pair(
    url: Annotated[
        str,
        typer.Argument(
            help="how your phone reaches the server, e.g. https://photos.tailnet-name.ts.net"
        ),
    ],
) -> None:
    """Show a link + QR code that logs a phone in (the token rides in the URL fragment,
    which browsers never send to the server or put in logs)."""
    from api.config import get_settings

    tok = get_settings().auth_token
    if not tok:
        console.print("PS_AUTH_TOKEN is not set; generate one with `photo-search token`")
        raise typer.Exit(1)
    link = f"{url.rstrip('/')}/#token={tok}"
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(link)
        qr.print_ascii(invert=True)
    except ImportError:
        pass
    console.print(link)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the API (and the built web UI, if present)."""
    import uvicorn

    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


@app.command()
def worker(queues: Annotated[list[str] | None, typer.Argument()] = None) -> None:
    """Run an RQ worker (queues: ingest, gpu)."""
    from workers.queue import run_worker

    run_worker(queues or ["ingest", "gpu"])


@app.command()
def watch(
    paths: Annotated[list[Path] | None, typer.Argument()] = None,
    inline: Annotated[
        bool, typer.Option(help="index in this process instead of the Redis queue")
    ] = False,
    poll: Annotated[
        bool, typer.Option(help="poll instead of inotify (network drives, /mnt/c on WSL)")
    ] = False,
) -> None:
    """Watch folders (and PS_IMPORT_DIRS) and index new/changed photos as they appear."""
    from workers.watch import watch as do_watch

    do_watch(paths or None, inline=inline, poll=poll)


@app.command("import")
def import_cmd(
    dirs: Annotated[list[Path] | None, typer.Argument(help="default: PS_IMPORT_DIRS")] = None,
    move: Annotated[bool, typer.Option(help="move instead of copy")] = False,
    no_index: bool = False,
) -> None:
    """Import new photos from camera-roll backup folders into the library, then index them."""
    from api.config import get_settings
    from workers.importer import import_and_index

    if move:
        get_settings().import_mode = "move"
    console.print(import_and_index(dirs or None, index=not no_index))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
