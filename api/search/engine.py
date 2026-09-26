"""The retrieval core shared by the API, the eval runner and the agent.

    query --parse--> semantic text + filters
          --encode--> CLIP text vector, caption-space vector, tsquery
          --retrieve--> one ranked candidate list per signal, filters applied in SQL
          --fuse--> weighted RRF
          --hydrate--> photo rows (video frames collapsed to their video)

Signals:
    clip     CLIP text -> image embedding similarity
    caption  sentence embedding of the query vs. VLM caption embeddings
    keyword  full text over captions and Lightroom keywords
    ocr      full text over OCR'd text in the image
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import uuid
from functools import lru_cache
from typing import Any

import numpy as np
from psycopg import sql
from pydantic import BaseModel, Field

from api.config import get_settings
from api.db.session import Conn, get_conn
from api.db.util import vec
from api.search import filters as flt
from api.search.fusion import rrf
from api.search.query_parser import Mode, ParsedQuery, parse_query
from api.urls import display_url, thumb_url

SIGNALS = ("clip", "caption", "keyword", "ocr")
# Filtered queries matching fewer photos than this are ranked exactly (no ANN):
# a brute-force scan of 20k vectors is a few ms and never loses results to HNSW
# post-filtering.
EXACT_SCAN_MAX = 20_000
ROCCHIO = (1.0, 0.75, 0.25)  # alpha, beta, gamma


class SearchRequest(BaseModel):
    q: str = ""
    filters: flt.SearchFilters | None = Field(
        None, description="explicit filters (e.g. edited chips); combined with parsed ones"
    )
    parse: Mode = "auto"
    k: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)
    model: str | None = None
    signals: list[str] | None = None
    weights: dict[str, float] | None = None
    positive_ids: list[uuid.UUID] = Field(default_factory=list)
    negative_ids: list[uuid.UUID] = Field(default_factory=list)
    like_photo_id: uuid.UUID | None = None
    image_vector: list[float] | None = Field(None, exclude=True)
    group_videos: bool = True
    fallback: bool = True


class Hit(BaseModel):
    photo_id: uuid.UUID
    score: float
    ranks: dict[str, int] = Field(default_factory=dict)
    path: str
    file_hash: str
    thumb_url: str
    display_url: str = ""
    taken_at: Any = None
    place_name: str | None = None
    country: str | None = None
    camera: str | None = None
    lens: str | None = None
    focal_length: float | None = None
    aperture: float | None = None
    width: int | None = None
    height: int | None = None
    caption: str | None = None
    is_video_frame: bool = False
    video_id: uuid.UUID | None = None
    frame_ts: float | None = None


class SearchResponse(BaseModel):
    query: str
    parsed: ParsedQuery
    hits: list[Hit]
    signals: list[str]
    fallback_used: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------- helpers


def fusion_weights(override: dict[str, float] | None = None) -> dict[str, float]:
    s = get_settings()
    w = dict(s.fusion_weights)
    if s.fusion_weights_file.exists():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            w.update(json.loads(s.fusion_weights_file.read_text()).get("weights", {}))
    if override:
        w.update(override)
    return w


_avail_lock = threading.Lock()
_avail: tuple[float, dict[str, bool]] | None = None


def available_signals(conn: Conn) -> dict[str, bool]:
    """Which signals have any data. Cached briefly so we don't load the caption
    embedder for a library that has no captions yet."""
    global _avail
    with _avail_lock:
        if _avail and time.monotonic() - _avail[0] < 30:
            return _avail[1]
    row = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM captions) AS cap, "
        "EXISTS (SELECT 1 FROM ocr_text WHERE text <> '') AS ocr, "
        "EXISTS (SELECT 1 FROM photos WHERE keywords <> '{}') AS kw"
    ).fetchone()
    assert row is not None
    a = {"clip": True, "caption": row["cap"], "keyword": row["cap"] or row["kw"], "ocr": row["ocr"]}
    with _avail_lock:
        _avail = (time.monotonic(), a)
    return a


def reset_caches() -> None:
    global _avail
    with _avail_lock:
        _avail = None
    _pgvector_version.cache_clear()


@lru_cache(maxsize=1)
def _pgvector_version() -> tuple[int, ...]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    return tuple(int(x) for x in row["extversion"].split(".")) if row else (0,)


def _count_filtered(conn: Conn, where: str, params: dict) -> int:
    row = conn.execute(f"SELECT count(*) AS n FROM photos p WHERE TRUE {where}", params).fetchone()
    return int(row["n"]) if row else 0


def _ann_settings(conn: Conn, exact: bool, n: int) -> None:
    ef = max(get_settings().hnsw_ef_search, n)
    conn.execute(sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(min(ef, 1000))))
    if not exact and _pgvector_version() >= (0, 8):
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")


def _vector_ranked(
    conn: Conn,
    *,
    table_sql: str,
    dist_sql: str,
    where: str,
    params: dict,
    n: int,
    exact: bool,
) -> list[tuple[uuid.UUID, float]]:
    """Top-n (photo_id, similarity). `exact` fences the ORDER BY behind OFFSET 0 so
    the planner can't use the HNSW index (a filtered ANN scan can silently return
    fewer than n rows)."""
    inner = f"SELECT x.photo_id, 1 - ({dist_sql}) AS sim FROM {table_sql} WHERE TRUE {where}"
    if exact:
        q = f"SELECT photo_id, sim FROM ({inner} OFFSET 0) s ORDER BY sim DESC LIMIT %(n)s"
    else:
        q = f"{inner} ORDER BY {dist_sql} LIMIT %(n)s"
    rows = conn.execute(q, {**params, "n": n}).fetchall()
    seen: dict[uuid.UUID, float] = {}
    for r in rows:
        seen.setdefault(r["photo_id"], float(r["sim"]))
    return list(seen.items())


def clip_candidates(
    conn: Conn, qvec: np.ndarray, model: str, dim: int, where: str, params: dict,
    n: int, exact: bool,
) -> list[tuple[uuid.UUID, float]]:  # fmt: skip
    dist = f"(x.embedding::vector({int(dim)})) <=> %(qvec)s::vector({int(dim)})"
    table = (
        "image_embeddings x JOIN photos p ON p.id = x.photo_id "
        f"AND x.model = {sql.Literal(model).as_string(conn)}"
    )
    return _vector_ranked(
        conn, table_sql=table, dist_sql=dist, where=where,
        params={**params, "qvec": qvec}, n=n, exact=exact,
    )  # fmt: skip


def caption_candidates(
    conn: Conn, cvec: np.ndarray, where: str, params: dict, n: int, exact: bool
) -> list[tuple[uuid.UUID, float]]:
    dist = "x.caption_embedding <=> %(cvec)s::vector(384)"
    table = (
        "captions x JOIN photos p ON p.id = x.photo_id AND x.caption_embedding IS NOT NULL "
        "AND x.embed_model = %(embed_model)s"
    )
    return _vector_ranked(
        conn, table_sql=table, dist_sql=dist, where=where,
        params={**params, "cvec": cvec, "embed_model": get_settings().text_embed_model},
        n=n, exact=exact,
    )  # fmt: skip


# OR-semantics tsquery: captions rarely contain every query word, and ts_rank_cd
# already rewards documents that match more of them.
_OR_TSQUERY = "nullif(replace(plainto_tsquery('english', %(kq)s)::text, '&', '|'), '')::tsquery"


def keyword_candidates(
    conn: Conn, text: str, where: str, params: dict, n: int
) -> list[tuple[uuid.UUID, float]]:
    rows = conn.execute(
        f"""
        WITH q AS (SELECT {_OR_TSQUERY} AS tsq),
        hits AS (
            SELECT c.photo_id, ts_rank_cd(c.tsv, q.tsq) AS score FROM captions c, q
            WHERE q.tsq IS NOT NULL AND c.tsv @@ q.tsq
            UNION ALL
            SELECT p.id, 0.5 * ts_rank_cd(p.keywords_tsv, q.tsq) FROM photos p, q
            WHERE q.tsq IS NOT NULL AND p.keywords_tsv @@ q.tsq
        )
        SELECT h.photo_id, sum(h.score) AS score FROM hits h JOIN photos p ON p.id = h.photo_id
        WHERE TRUE {where}
        GROUP BY h.photo_id ORDER BY score DESC, h.photo_id LIMIT %(n)s
        """,
        {**params, "kq": text, "n": n},
    ).fetchall()
    return [(r["photo_id"], float(r["score"])) for r in rows]


def ocr_candidates(
    conn: Conn, text: str, where: str, params: dict, n: int
) -> list[tuple[uuid.UUID, float]]:
    rows = conn.execute(
        f"""
        WITH q AS (SELECT {_OR_TSQUERY} AS tsq)
        SELECT o.photo_id, ts_rank_cd(o.tsv, q.tsq) AS score
        FROM ocr_text o JOIN photos p ON p.id = o.photo_id, q
        WHERE q.tsq IS NOT NULL AND o.tsv @@ q.tsq {where}
        ORDER BY score DESC, o.photo_id LIMIT %(n)s
        """,
        {**params, "kq": text, "n": n},
    ).fetchall()
    return [(r["photo_id"], float(r["score"])) for r in rows]


def photo_vectors(conn: Conn, ids: list[uuid.UUID], model: str) -> np.ndarray:
    if not ids:
        return np.zeros((0, 0), np.float32)
    rows = conn.execute(
        "SELECT embedding FROM image_embeddings WHERE photo_id = ANY(%s) AND model = %s",
        (ids, model),
    ).fetchall()
    if not rows:
        return np.zeros((0, 0), np.float32)
    return np.stack([vec(r["embedding"]) for r in rows])


def rocchio(q: np.ndarray | None, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    a, b, g = ROCCHIO
    v = a * q if q is not None else np.zeros(pos.shape[1] if len(pos) else neg.shape[1], np.float32)
    if len(pos):
        v = v + b * pos.mean(0)
    if len(neg):
        v = v - g * neg.mean(0)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n else v.astype(np.float32)


HYDRATE_SQL = """
SELECT p.id, p.path, p.file_hash, p.taken_at, p.place_name, p.country, p.camera, p.lens,
       p.focal_length, p.aperture, p.width, p.height, p.is_video_frame, p.source_video_id,
       p.frame_ts, v.path AS video_path,
       (SELECT caption FROM captions c WHERE c.photo_id = p.id ORDER BY c.created_at DESC LIMIT 1)
         AS caption
FROM photos p LEFT JOIN photos v ON v.id = p.source_video_id
WHERE p.id = ANY(%s)
"""


def hydrate(
    conn: Conn,
    fused: list[tuple[Any, float, dict[str, int]]],
    *,
    k: int,
    offset: int,
    group_videos: bool,
) -> list[Hit]:
    want = (k + offset) * (3 if group_videos else 1)
    top = fused[:want]
    rows = {r["id"]: r for r in conn.execute(HYDRATE_SQL, ([pid for pid, _, _ in top],)).fetchall()}
    hits: list[Hit] = []
    seen_videos: set[uuid.UUID] = set()
    for pid, score, ranks in top:
        r = rows.get(pid)
        if r is None:
            continue
        if group_videos and r["is_video_frame"]:
            if r["source_video_id"] in seen_videos:
                continue
            seen_videos.add(r["source_video_id"])
        hits.append(
            Hit(
                photo_id=pid,
                score=score,
                ranks=ranks,
                path=r["video_path"] or r["path"],
                file_hash=r["file_hash"],
                thumb_url=thumb_url(pid, r["file_hash"]),
                display_url=display_url(pid, r["file_hash"]),
                taken_at=r["taken_at"],
                place_name=r["place_name"],
                country=r["country"],
                camera=r["camera"],
                lens=r["lens"],
                focal_length=r["focal_length"],
                aperture=r["aperture"],
                width=r["width"],
                height=r["height"],
                caption=r["caption"],
                is_video_frame=r["is_video_frame"],
                video_id=r["source_video_id"],
                frame_ts=r["frame_ts"],
            )
        )
    return hits[offset : offset + k]


def merge_filters(a: flt.SearchFilters, b: flt.SearchFilters | None) -> flt.SearchFilters:
    """b (explicit, user-edited) wins field by field over a (parsed)."""
    if b is None:
        return a
    merged = a.model_dump()
    for k, v in b.model_dump().items():
        if v not in (None, []):
            merged[k] = v
    return flt.SearchFilters(**merged)


# ---------------------------------------------------------------- entry point


def search(req: SearchRequest, conn: Conn | None = None) -> SearchResponse:
    if conn is None:
        with get_conn() as c:
            return search(req, c)

    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(name: str, t0: float) -> float:
        now = time.perf_counter()
        timings[name] = round((now - t0) * 1000, 2)
        return now

    t = time.perf_counter()
    by_example = req.like_photo_id is not None or req.image_vector is not None
    if by_example or req.parse == "off":
        parsed = ParsedQuery(text=req.q, semantic=req.q, filters=flt.SearchFilters(), source="none")
    else:
        parsed = parse_query(req.q, req.parse)
    if req.filters is not None:
        parsed.filters = merge_filters(parsed.filters, req.filters)
        if parsed.source == "none":
            parsed.source = "user"
    t = lap("parse", t)

    resp = _retrieve(conn, req, parsed, by_example, timings)
    if not resp.hits and req.fallback and not parsed.filters.is_empty() and not by_example:
        # parser (or user) produced filters that match nothing: fall back to pure semantic
        loose = ParsedQuery(
            text=req.q, semantic=req.q, filters=flt.SearchFilters(), source=parsed.source,
            error=parsed.error,
        )  # fmt: skip
        resp = _retrieve(conn, req, loose, by_example, timings)
        resp.parsed = parsed
        resp.fallback_used = True
    timings["total"] = round((time.perf_counter() - t_start) * 1000, 2)
    resp.timings_ms = timings
    return resp


def _retrieve(
    conn: Conn,
    req: SearchRequest,
    parsed: ParsedQuery,
    by_example: bool,
    timings: dict[str, float],
) -> SearchResponse:
    rankings = compute_rankings(conn, req, parsed, by_example, timings)
    t = time.perf_counter()
    fused = rrf(rankings, fusion_weights(req.weights) | {"recency": 1.0}, k=get_settings().rrf_k)
    hits = hydrate(conn, fused, k=req.k, offset=req.offset, group_videos=req.group_videos)
    timings["fuse_hydrate"] = round((time.perf_counter() - t) * 1000, 2)
    return SearchResponse(query=req.q, parsed=parsed, hits=hits, signals=list(rankings))


def compute_rankings(
    conn: Conn,
    req: SearchRequest,
    parsed: ParsedQuery,
    by_example: bool = False,
    timings: dict[str, float] | None = None,
) -> dict[str, list[uuid.UUID]]:
    """One ranked candidate list per signal, filters already applied. Exposed so
    fusion-weight tuning can re-fuse cached rankings without re-querying."""
    from api.ml.clip import get_clip
    from api.ml.text_embed import get_text_embedder

    timings = timings if timings is not None else {}
    enc = get_clip(req.model)
    model = enc.spec.name
    where, params = flt.to_sql(parsed.filters)
    exclude: list[uuid.UUID] = list(req.negative_ids)
    if req.like_photo_id:
        exclude.append(req.like_photo_id)
    if exclude:
        where += " AND NOT (p.id = ANY(%(x_exclude)s))"
        params["x_exclude"] = exclude
    where += " AND p.media_type = 'image' AND p.error IS NULL"
    n = max(200, (req.k + req.offset) * 4)
    semantic = parsed.semantic.strip()

    avail = available_signals(conn)
    wanted = [sig for sig in (req.signals or SIGNALS) if sig in SIGNALS and avail.get(sig)]
    if by_example:
        wanted = ["clip"]
    elif not semantic and not req.positive_ids:
        wanted = []

    t = time.perf_counter()
    qvec: np.ndarray | None = None
    if "clip" in wanted:
        if req.image_vector is not None:
            qvec = np.asarray(req.image_vector, np.float32)
        elif req.like_photo_id is not None:
            v = photo_vectors(conn, [req.like_photo_id], model)
            qvec = v[0] if len(v) else None
        elif semantic:
            qvec = enc.encode_texts([semantic])[0]
        if req.positive_ids or req.negative_ids:
            qvec = rocchio(
                qvec,
                photo_vectors(conn, req.positive_ids, model),
                photo_vectors(conn, req.negative_ids, model),
            )
    cvec = None
    if "caption" in wanted and semantic:
        cvec = get_text_embedder().encode_query(semantic)
    timings["encode"] = round((time.perf_counter() - t) * 1000, 2)

    t = time.perf_counter()
    exact = False
    if not parsed.filters.is_empty():
        exact = _count_filtered(conn, where, params) <= EXACT_SCAN_MAX
    _ann_settings(conn, exact, n)
    rankings: dict[str, list[uuid.UUID]] = {}
    if qvec is not None:
        rankings["clip"] = [
            pid for pid, _ in clip_candidates(conn, qvec, model, enc.dim, where, params, n, exact)
        ]
    if cvec is not None:
        rankings["caption"] = [
            pid for pid, _ in caption_candidates(conn, cvec, where, params, n, exact)
        ]
    if "keyword" in wanted and semantic:
        rankings["keyword"] = [
            pid for pid, _ in keyword_candidates(conn, semantic, where, params, n)
        ]
    if "ocr" in wanted and semantic:
        rankings["ocr"] = [pid for pid, _ in ocr_candidates(conn, semantic, where, params, n)]
    if not wanted:
        # metadata-only query ("photos from Chicago in 2025"): newest first
        rows = conn.execute(
            f"SELECT p.id FROM photos p WHERE TRUE {where} ORDER BY p.taken_at DESC NULLS LAST, p.id "
            "LIMIT %(n)s",
            {**params, "n": n},
        ).fetchall()
        rankings["recency"] = [r["id"] for r in rows]
    timings["retrieve"] = round((time.perf_counter() - t) * 1000, 2)
    return rankings
