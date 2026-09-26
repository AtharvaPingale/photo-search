"""Labelling tool backend: write eval queries and their relevant photos.

Candidates are *pooled* from several retrieval systems (CLIP alone, captions,
keywords, OCR, the fused ranking, plus any extra phrasings the labeller
tries), so the labels aren't biased toward whatever the current best system
already finds. Unpooled labelling makes every later "improvement" look worse.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.urls import thumb_url
from eval import dataset
from eval.tracking import REPORTS_DIR

router = APIRouter(prefix="/eval", tags=["eval"])


class QueryIn(BaseModel):
    id: str | None = None
    query: str
    category: str = "objects"
    relevant: list[str] = Field(default_factory=list)
    grades: dict[str, int] = Field(default_factory=dict)
    notes: str = ""


class CandidatesIn(BaseModel):
    query: str
    extra_queries: list[str] = Field(default_factory=list)
    per_system: int = Field(30, ge=5, le=200)
    relevant: list[str] = Field(default_factory=list)


@router.get("/queries")
def list_queries() -> dict[str, Any]:
    qs = dataset.load_queries(labeled_only=False)
    return {
        "categories": dataset.CATEGORIES,
        "queries": [q.model_dump() for q in qs],
        "counts": {
            "total": len(qs),
            "labeled": sum(q.labeled for q in qs),
            "dev": sum(q.split == "dev" and q.labeled for q in qs),
            "test": sum(q.split == "test" and q.labeled for q in qs),
        },
    }


@router.post("/queries")
def save_query(q: QueryIn) -> dict[str, Any]:
    if q.category not in dataset.CATEGORIES:
        raise HTTPException(400, f"category must be one of {dataset.CATEGORIES}")
    saved = dataset.upsert(dataset.EvalQuery(id=q.id or "", **q.model_dump(exclude={"id"})))
    return saved.model_dump()


@router.delete("/queries/{qid}")
def delete_query(qid: str) -> dict[str, bool]:
    if not dataset.delete(qid):
        raise HTTPException(404, "no such query")
    return {"deleted": True}


@router.post("/candidates")
def candidates(body: CandidatesIn) -> dict[str, Any]:
    from api.search.engine import SearchRequest, search

    pool: dict[str, dict[str, Any]] = {}
    systems = [
        ("clip", dict(signals=["clip"], parse="off")),
        ("clip+parser", dict(signals=["clip"], parse="auto")),
        ("caption", dict(signals=["caption"], parse="auto")),
        ("keyword", dict(signals=["keyword", "ocr"], parse="auto")),
        ("fused", dict(parse="auto")),
    ]
    for text in [body.query, *body.extra_queries]:
        for name, kw in systems:
            resp = search(SearchRequest(q=text, k=body.per_system, group_videos=False, **kw))  # type: ignore[arg-type]
            for rank, h in enumerate(resp.hits, 1):
                c = pool.setdefault(h.file_hash, {**h.model_dump(mode="json"), "found_by": {}})
                c["found_by"][f"{name}:{text}" if text != body.query else name] = rank
    # labelled photos stay visible even if no system finds them any more
    missing = [h for h in body.relevant if h not in pool]
    if missing:
        from api.db.session import get_conn

        with get_conn() as conn:
            for r in conn.execute(
                "SELECT id, file_hash, path FROM photos WHERE file_hash = ANY(%s)", (missing,)
            ):
                pool[r["file_hash"]] = {
                    "photo_id": str(r["id"]), "file_hash": r["file_hash"], "path": r["path"],
                    "thumb_url": thumb_url(r["id"], r["file_hash"]), "found_by": {},
                }  # fmt: skip
    items = sorted(pool.values(), key=lambda c: min(c["found_by"].values(), default=10_000))
    return {"candidates": items, "n": len(items)}


@router.get("/reports")
def reports() -> list[dict[str, Any]]:
    out = []
    for p in sorted(REPORTS_DIR.glob("*.json"), reverse=True)[:50]:
        try:
            r = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out.append(
            {
                "file": p.name,
                "kind": r.get("kind", "retrieval"),
                "name": r.get("name"),
                "split": r.get("config", {}).get("split"),
                "created_at": r.get("created_at"),
                "git_commit": r.get("git_commit"),
                "overall": r.get("overall"),
                "latency_ms": r.get("latency_ms"),
            }
        )
    return out


@router.get("/reports/{name}")
def report(name: str) -> dict[str, Any]:
    p = (REPORTS_DIR / name).resolve()
    if p.parent != REPORTS_DIR.resolve() or not p.exists() or p.suffix != ".json":
        raise HTTPException(404, "no such report")
    return json.loads(p.read_text())
