"""FastAPI service: search, photos, people, albums, organisation, eval labelling, agent.

Endpoints that run model inference or SQL are plain `def`: FastAPI runs them
on its threadpool, which is the right shape for blocking GPU/DB work. The
built React UI (web/dist) is served from / when present.
"""

from __future__ import annotations

import logging
import mimetypes
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.auth import AuthMiddleware
from api.auth import router as auth_router
from api.config import ROOT
from api.routes import agent, albums, labels, organize, people, photos, search

log = logging.getLogger(__name__)
WEB_DIST = ROOT / "web" / "dist"
mimetypes.add_type("application/manifest+json", ".webmanifest")


@asynccontextmanager
async def lifespan(_: FastAPI):
    from api.db.session import close_pool, get_pool

    get_pool()
    # Load the text encoders now, in the background, so the first search from the
    # phone doesn't wait several seconds for model loading.
    threading.Thread(target=_warm_models, daemon=True).start()
    yield
    close_pool()


def _warm_models() -> None:
    try:
        from api.ml.clip import get_clip
        from api.ml.text_embed import get_text_embedder

        get_clip().encode_texts(["warm up"])
        get_text_embedder().encode_query("warm up")
    except Exception:  # a missing model shouldn't stop the API; the first search will report it
        log.exception("model warm-up failed")


def create_app() -> FastAPI:
    app = FastAPI(title="Photo Search", version="0.1.0", lifespan=lifespan)
    # the last middleware added runs first: CORS (answers preflights), then gzip, then auth
    app.add_middleware(AuthMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    app.include_router(auth_router, prefix="/api")
    for r in (search, photos, people, albums, organize, labels, agent):
        app.include_router(r.router, prefix="/api")

    @app.get("/api/health")
    def health() -> dict:
        from api.db.session import get_conn

        with get_conn() as conn:
            conn.execute("SELECT 1")
        return {"ok": True}

    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404, "no such endpoint")
            f = (WEB_DIST / path).resolve()
            if path and f.is_file() and WEB_DIST.resolve() in f.parents:
                # the service worker and manifest must always be revalidated
                no_cache = (
                    {"Cache-Control": "no-cache"}
                    if f.name in ("sw.js", "index.html", "manifest.webmanifest")
                    else {}
                )
                return FileResponse(f, headers=no_cache)
            return FileResponse(WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
