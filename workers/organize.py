"""Library organisation: near-duplicates, bursts, auto albums.

Near-duplicates need two independent signals to agree: CLIP cosine >= 0.95
(same content) *and* perceptual-hash distance <= 10 bits (same pixels, give or
take resizing and recompression). CLIP alone merges different shots of the
same scene; pHash alone is fooled by flat images.

Bursts are consecutive frames from one camera within a couple of seconds that
also look alike; the suggested pick is the sharpest (variance of Laplacian).

Auto albums segment the timeline at long gaps, label segments by place,
merge consecutive away-from-home segments into trips, and ask the LLM for a
title and one-line summary (with a template fallback).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from api.db.session import get_conn
from api.ml.clip import get_clip
from workers.embed import load_embeddings

log = logging.getLogger(__name__)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Any, Any] = {}

    def find(self, x: Any) -> Any:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: Any, b: Any) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb, key=str)] = min(ra, rb, key=str)

    def groups(self) -> list[list[Any]]:
        out: dict[Any, list[Any]] = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return [g for g in out.values() if len(g) > 1]


def topk_neighbours(X: np.ndarray, k: int = 10, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine kNN (rows L2-normalised) in chunks: 50k x 50k is fine on GPU or CPU."""
    k = min(k + 1, len(X))
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        T = torch.from_numpy(X).to(dev)
        sims, idx = [], []
        for i in range(0, len(X), chunk):
            s = T[i : i + chunk] @ T.T
            v, j = torch.topk(s, k, dim=1)
            sims.append(v.cpu().numpy())
            idx.append(j.cpu().numpy())
        return np.concatenate(sims), np.concatenate(idx)
    except ImportError:  # pragma: no cover
        S = X @ X.T
        top = np.argsort(-S, axis=1)[:, :k]
        return np.take_along_axis(S, top, 1), top


def hamming_hex(a: str | None, b: str | None) -> int:
    if not a or not b:
        return 64
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _write_groups(
    kind: str,
    groups: list[list[uuid.UUID]],
    scores: dict[uuid.UUID, float],
    best: dict[int, uuid.UUID],
) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM photo_groups WHERE kind = %s", (kind,))
        cur.executemany(
            "INSERT INTO photo_groups (kind, group_id, photo_id, score, is_best) VALUES (%s, %s, %s, %s, %s)",
            [
                (kind, gi, pid, scores.get(pid), best.get(gi) == pid)
                for gi, g in enumerate(groups)
                for pid in g
            ],
        )
        conn.commit()


def find_duplicates(
    sim_threshold: float = 0.95, phash_max: int = 10, model: str | None = None
) -> dict[str, int]:
    model = model or get_clip().spec.name
    ids, X = load_embeddings(model, "AND NOT p.is_video_frame")
    if len(ids) < 2:
        _write_groups("duplicate", [], {}, {})
        return {"groups": 0, "photos": 0}
    with get_conn() as conn:
        meta = {
            r["id"]: r
            for r in conn.execute(
                "SELECT id, phash, width, height, sharpness FROM photos WHERE id = ANY(%s)", (ids,)
            )
        }
    sims, idx = topk_neighbours(X, k=10)
    uf = UnionFind()
    for i, (srow, irow) in enumerate(zip(sims, idx, strict=True)):
        for s, j in zip(srow, irow, strict=True):
            if j == i or s < sim_threshold:
                continue
            a, b = ids[i], ids[j]
            if hamming_hex(meta[a]["phash"], meta[b]["phash"]) <= phash_max:
                uf.union(a, b)
    groups = uf.groups()
    # best copy: most pixels, then sharpest
    best = {
        gi: max(
            g,
            key=lambda p: (
                (meta[p]["width"] or 0) * (meta[p]["height"] or 0),
                meta[p]["sharpness"] or 0,
            ),
        )
        for gi, g in enumerate(groups)
    }
    _write_groups("duplicate", groups, {}, best)
    return {"groups": len(groups), "photos": sum(len(g) for g in groups)}


def find_bursts(
    max_gap_s: float = 2.0, sim_threshold: float = 0.85, model: str | None = None
) -> dict[str, int]:
    model = model or get_clip().spec.name
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT p.id, p.camera, p.taken_at, p.sharpness, e.embedding FROM photos p "
            "JOIN image_embeddings e ON e.photo_id = p.id AND e.model = %s "
            "WHERE NOT p.is_video_frame AND p.taken_at IS NOT NULL ORDER BY p.camera, p.taken_at, p.id",
            (model,),
        ).fetchall()
    from api.db.util import vec

    groups: list[list[uuid.UUID]] = []
    cur: list[dict] = []
    for r in rows:
        if cur:
            prev = cur[-1]
            same = (
                prev["camera"] == r["camera"]
                and (r["taken_at"] - prev["taken_at"]).total_seconds() <= max_gap_s
                and float(vec(prev["embedding"]) @ vec(r["embedding"])) >= sim_threshold
            )
            if not same:
                if len(cur) > 1:
                    groups.append([x["id"] for x in cur])
                cur = []
        cur.append(r)
    if len(cur) > 1:
        groups.append([x["id"] for x in cur])
    sharp = {r["id"]: r["sharpness"] or 0.0 for r in rows}
    best = {gi: max(g, key=lambda p: sharp[p]) for gi, g in enumerate(groups)}
    _write_groups("burst", groups, sharp, best)
    return {"groups": len(groups), "photos": sum(len(g) for g in groups)}


# ------------------------------------------------------------------ albums


@dataclass
class Segment:
    photos: list[dict] = field(default_factory=list)

    @property
    def start(self) -> datetime:
        return self.photos[0]["taken_at"]

    @property
    def end(self) -> datetime:
        return self.photos[-1]["taken_at"]

    def place(self) -> str | None:
        c = Counter(p["place_name"] for p in self.photos if p["place_name"])
        return c.most_common(1)[0][0] if c else None

    def region(self) -> str | None:
        c = Counter((p["admin1"], p["country"]) for p in self.photos if p["country"])
        return "|".join(x or "" for x in c.most_common(1)[0][0]) if c else None


def segment_timeline(photos: list[dict], gap_h: float = 8.0) -> list[Segment]:
    segs: list[Segment] = []
    for p in photos:
        if segs and (p["taken_at"] - segs[-1].end) <= timedelta(hours=gap_h):
            segs[-1].photos.append(p)
        else:
            segs.append(Segment([p]))
    return segs


def home_region(photos: list[dict]) -> str | None:
    """Where most photo-days happen is home."""
    days: Counter[str] = Counter()
    seen = set()
    for p in photos:
        if not p["country"]:
            continue
        key = (p["taken_at"].date(), p["admin1"], p["country"])
        if key not in seen:
            seen.add(key)
            days[f"{p['admin1'] or ''}|{p['country']}"] += 1
    return days.most_common(1)[0][0] if days else None


def group_trips(segs: list[Segment], home: str | None, max_gap_h: float = 48.0) -> list[Segment]:
    """Merge consecutive segments away from home into one trip."""
    out: list[Segment] = []
    for s in segs:
        away = s.region() is not None and s.region() != home
        if (
            out
            and away
            and getattr(out[-1], "_away", False)
            and (s.start - out[-1].end) <= timedelta(hours=max_gap_h)
        ):
            out[-1].photos.extend(s.photos)
        else:
            s._away = away  # type: ignore[attr-defined]
            out.append(s)
    return out


def _title_fallback(seg: Segment) -> str:
    place = seg.place() or "Photos"
    a, b = seg.start, seg.end
    if a.date() == b.date():
        return f"{place} · {a:%b} {a.day}, {a:%Y}"
    if (a.year, a.month) == (b.year, b.month):
        return f"{place} · {a:%b} {a.day}–{b.day}, {a:%Y}"
    return f"{place} · {a:%b %Y}" + (
        f" – {b:%b %Y}" if (a.year, a.month) != (b.year, b.month) else ""
    )


def _llm_title(seg: Segment, captions: list[str]) -> tuple[str, str] | None:
    from api.llm import LLMError, complete_json, llm_enabled

    if not llm_enabled():
        return None
    places = Counter(p["place_name"] for p in seg.photos if p["place_name"]).most_common(5)
    prompt = (
        f"Dates: {seg.start:%Y-%m-%d} to {seg.end:%Y-%m-%d} ({len(seg.photos)} photos)\n"
        f"Places: {', '.join(f'{n} ({c})' for n, c in places) or 'unknown'}\n"
        "Sample photo descriptions:\n" + "\n".join(f"- {c}" for c in captions[:15])
    )
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}, "summary": {"type": "string"}},
        "required": ["title", "summary"],
        "additionalProperties": False,
    }
    try:
        out = complete_json(
            "You title photo albums. Title: at most 6 words, specific (place + what happened), no emoji, "
            "no quotes. Summary: one plain sentence. Only use facts present in the input.",
            prompt,
            schema,
            max_tokens=300,
        )
        return out["title"].strip()[:80], out["summary"].strip()[:400]
    except (LLMError, KeyError) as e:
        log.info("album title LLM failed, using template: %s", e)
        return None


def build_albums(
    min_photos: int = 15, gap_h: float = 8.0, llm_titles: bool = True
) -> dict[str, int]:
    with get_conn() as conn:
        photos = conn.execute(
            "SELECT id, taken_at, place_name, admin1, country, lat, lon, sharpness FROM photos "
            "WHERE media_type = 'image' AND NOT is_video_frame AND taken_at IS NOT NULL AND error IS NULL "
            "ORDER BY taken_at, id"
        ).fetchall()
    segs = group_trips(segment_timeline(photos, gap_h), home_region(photos))
    albums = [s for s in segs if len(s.photos) >= min_photos]

    with get_conn() as conn:
        conn.execute("DELETE FROM albums WHERE auto")
        conn.commit()
    for seg in albums:
        ids = [p["id"] for p in seg.photos]
        with get_conn() as conn:
            caps = [
                r["caption"]
                for r in conn.execute(
                    "SELECT DISTINCT ON (photo_id) caption FROM captions WHERE photo_id = ANY(%s) LIMIT 40",
                    (ids,),
                )
            ]
        got = _llm_title(seg, caps) if llm_titles else None
        title, summary = got or (_title_fallback(seg), None)
        cover = max(seg.photos, key=lambda p: p["sharpness"] or 0)["id"]
        lat = np.nanmean([p["lat"] for p in seg.photos if p["lat"] is not None] or [np.nan])
        lon = np.nanmean([p["lon"] for p in seg.photos if p["lon"] is not None] or [np.nan])
        aid = uuid.uuid4()
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO albums (id, title, summary, start_at, end_at, place_name, lat, lon, cover_photo_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (aid, title, summary, seg.start, seg.end, seg.place(),
                 None if np.isnan(lat) else float(lat), None if np.isnan(lon) else float(lon), cover),
            )  # fmt: skip
            cur.executemany(
                "INSERT INTO album_photos (album_id, photo_id) VALUES (%s, %s)",
                [(aid, i) for i in ids],
            )
            conn.commit()
    return {"albums": len(albums), "segments": len(segs)}


def albums_json() -> str:  # debugging aid
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT title, start_at, end_at, place_name FROM albums ORDER BY start_at"
        ).fetchall()
    return json.dumps(rows, default=str, indent=2)
