"""Agent tools. The model never writes SQL.

  semantic_search(query, filters, sort, limit)  -> the normal search engine
  metadata_query(template, params)              -> one of a fixed set of named,
                                                   parameterised SQL templates
  aggregate(group_by, filters, top_n)           -> counts grouped by a whitelisted dimension
  get_photo_details(photo_id)                   -> one photo's metadata, caption, OCR, people

Every argument is validated with pydantic before it reaches SQL; group_by and
template names map to fixed SQL fragments in this file; values are always bound
parameters. Tool results are compact JSON (<= 20 photos) and every photo id a
tool returns is recorded, so the final answer's citations can be checked.
"""

from __future__ import annotations

import json
import uuid
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from api.db.session import get_conn, one
from api.search.filters import SearchFilters, to_sql

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover

    def traceable(*_a: Any, **_k: Any):  # type: ignore[no-redef]
        return lambda f: f


MAX_PHOTOS = 20


def _photo_brief(r: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    out = {
        "photo_id": str(r["id"] if "id" in r else r["photo_id"]),
        "taken_at": str(r["taken_at"])[:16] if r.get("taken_at") else None,
        "place": ", ".join(x for x in (r.get("place_name"), r.get("country")) if x) or None,
        "camera": r.get("camera"),
        "lens": r.get("lens"),
    }
    if r.get("caption"):
        out["caption"] = r["caption"][:160]
    if score is not None:
        out["relevance"] = round(score, 4)
    return {k: v for k, v in out.items() if v is not None}


# ------------------------------------------------------------------ semantic_search


class SemanticSearchArgs(BaseModel):
    query: str = Field(description="what the photos show, e.g. 'Columbus skyline at night'")
    filters: SearchFilters = Field(default_factory=SearchFilters)
    sort: Literal["relevance", "newest", "oldest"] = "relevance"
    limit: int = Field(10, ge=1, le=MAX_PHOTOS)


def semantic_search(args: SemanticSearchArgs) -> dict[str, Any]:
    from api.search.engine import SearchRequest, search

    # the agent passes structured filters itself, so the query parser is off
    resp = search(
        SearchRequest(
            q=args.query, filters=args.filters, parse="off", k=max(args.limit, 30), fallback=False
        )
    )
    hits = resp.hits
    if args.sort != "relevance":
        # "when did I LAST shoot X": the `limit` most relevant photos, newest (or oldest)
        # first. Date-sorting a longer list and then cutting it would drop the best
        # matches in favour of recent-but-irrelevant ones.
        head = hits[: args.limit]
        dated = sorted((h for h in head if h.taken_at is not None), key=lambda h: h.taken_at)
        undated = [h for h in head if h.taken_at is None]
        hits = (dated[::-1] if args.sort == "newest" else dated) + undated
    photos = [
        _photo_brief({**h.model_dump(), "id": h.photo_id}, h.score) for h in hits[: args.limit]
    ]
    return {"photos": photos, "note": "relevance is a fused rank score; higher is better"}


# ------------------------------------------------------------------ metadata_query


class DateRange(BaseModel):
    date_from: date | None = None
    date_to: date | None = None


class PhotosInRange(DateRange):
    place: str | None = None
    camera: str | None = None
    lens: str | None = None
    person: str | None = None
    limit: int = Field(20, ge=1, le=MAX_PHOTOS)


class FirstLast(BaseModel):
    place: str | None = None
    camera: str | None = None
    lens: str | None = None
    person: str | None = None


class OnDate(BaseModel):
    day: date


class Albums(DateRange):
    pass


class Library(BaseModel):
    pass


def _filters(p: Any) -> SearchFilters:
    return SearchFilters(
        date_from=getattr(p, "date_from", None),
        date_to=getattr(p, "date_to", None),
        place=getattr(p, "place", None),
        camera=getattr(p, "camera", None),
        lens=getattr(p, "lens", None),
        people=[p.person] if getattr(p, "person", None) else [],
    )


BRIEF_COLS = "p.id, p.taken_at, p.place_name, p.country, p.camera, p.lens"


def _t_photos_in_range(p: PhotosInRange) -> dict[str, Any]:
    where, params = to_sql(_filters(p))
    with get_conn() as conn:
        n = one(
            conn, f"SELECT count(*) AS n FROM photos p WHERE p.media_type = 'image' {where}", params
        )["n"]
        rows = conn.execute(
            f"SELECT {BRIEF_COLS} FROM photos p WHERE p.media_type = 'image' {where} "
            "ORDER BY p.taken_at DESC NULLS LAST LIMIT %(lim)s",
            {**params, "lim": p.limit},
        ).fetchall()
    return {"total_matching": n, "photos": [_photo_brief(r) for r in rows]}


def _t_first_last(p: FirstLast) -> dict[str, Any]:
    where, params = to_sql(_filters(p))
    with get_conn() as conn:
        first = conn.execute(
            f"SELECT {BRIEF_COLS} FROM photos p WHERE p.taken_at IS NOT NULL {where} ORDER BY p.taken_at LIMIT 1",
            params,
        ).fetchone()
        last = conn.execute(
            f"SELECT {BRIEF_COLS} FROM photos p WHERE p.taken_at IS NOT NULL {where} ORDER BY p.taken_at DESC LIMIT 1",
            params,
        ).fetchone()
        n = one(conn, f"SELECT count(*) AS n FROM photos p WHERE TRUE {where}", params)["n"]
    return {
        "total_matching": n,
        "first": _photo_brief(first) if first else None,
        "last": _photo_brief(last) if last else None,
    }


def _t_on_date(p: OnDate) -> dict[str, Any]:
    return _t_photos_in_range(PhotosInRange(date_from=p.day, date_to=p.day))


def _t_albums(p: Albums) -> dict[str, Any]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT a.id, a.title, a.summary, a.start_at, a.end_at, a.place_name, a.cover_photo_id, "
            "count(ap.photo_id) AS n FROM albums a LEFT JOIN album_photos ap ON ap.album_id = a.id "
            "WHERE (%(f)s::date IS NULL OR a.end_at >= %(f)s) AND (%(t)s::date IS NULL OR a.start_at < %(t)s::date + 1) "
            "GROUP BY a.id ORDER BY a.start_at LIMIT 50",
            {"f": p.date_from, "t": p.date_to},
        ).fetchall()
    return {
        "albums": [
            {
                "album_id": str(r["id"]), "title": r["title"], "summary": r["summary"],
                "start": str(r["start_at"])[:10], "end": str(r["end_at"])[:10],
                "place": r["place_name"], "n_photos": r["n"],
                "cover_photo_id": str(r["cover_photo_id"]) if r["cover_photo_id"] else None,
            }
            for r in rows
        ]
    }  # fmt: skip


def _t_library(_: Library) -> dict[str, Any]:
    with get_conn() as conn:
        r = one(
            conn,
            "SELECT count(*) FILTER (WHERE NOT is_video_frame AND media_type = 'image') AS photos, "
            "count(*) FILTER (WHERE media_type = 'video') AS videos, min(taken_at) AS first, "
            "max(taken_at) AS last, count(DISTINCT place_name) AS places, count(DISTINCT camera) AS cameras "
            "FROM photos",
        )
        people = [
            x["name"]
            for x in conn.execute("SELECT name FROM people WHERE name IS NOT NULL ORDER BY name")
        ]
    return {**{k: str(v) for k, v in r.items()}, "named_people": people}


TEMPLATES: dict[str, tuple[type[BaseModel], Any, str]] = {
    "photos_in_range": (PhotosInRange, _t_photos_in_range, "Photos matching date range / place / camera / lens / person, newest first, with a total count."),
    "first_and_last": (FirstLast, _t_first_last, "The first and the most recent photo matching place / camera / lens / person."),
    "photos_on_date": (OnDate, _t_on_date, "Photos taken on one day."),
    "albums_in_range": (Albums, _t_albums, "Auto albums (trips and events) overlapping a date range, with ids usable as filters.album_id."),
    "library_overview": (Library, _t_library, "Library size, date span, number of places and cameras, named people."),
}  # fmt: skip


class MetadataQueryArgs(BaseModel):
    template: Literal[
        "photos_in_range", "first_and_last", "photos_on_date", "albums_in_range", "library_overview"
    ]
    params: dict[str, Any] = Field(default_factory=dict)


def metadata_query(args: MetadataQueryArgs) -> dict[str, Any]:
    model, fn, _ = TEMPLATES[args.template]
    return fn(model(**args.params))


# ------------------------------------------------------------------ aggregate

GROUP_BY: dict[str, str] = {
    "city": "p.place_name",
    "region": "p.admin1",
    "country": "p.country",
    "camera": "p.camera",
    "lens": "p.lens",
    "year": "EXTRACT(YEAR FROM p.taken_at)::int::text",
    "month": "to_char(p.taken_at, 'YYYY-MM')",
    "weekday": "to_char(p.taken_at, 'FMDay')",
    "hour": "EXTRACT(HOUR FROM p.taken_at)::int::text",
    "focal_length": "round(p.focal_length)::int::text",
    "aperture": "p.aperture::text",
    "person": "(SELECT string_agg(DISTINCT pe.name, ', ') FROM faces f JOIN people pe ON pe.cluster_id = f.cluster_id WHERE f.photo_id = p.id AND pe.name IS NOT NULL)",
}


class AggregateArgs(BaseModel):
    group_by: Literal[
        "city",
        "region",
        "country",
        "camera",
        "lens",
        "year",
        "month",
        "weekday",
        "hour",
        "focal_length",
        "aperture",
        "person",
    ]
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_n: int = Field(10, ge=1, le=50)


def aggregate(args: AggregateArgs) -> dict[str, Any]:
    expr = GROUP_BY[args.group_by]
    where, params = to_sql(args.filters)
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT g AS value, count(*) AS n, (array_agg(id ORDER BY sharpness DESC NULLS LAST))[1] AS example
            FROM (SELECT p.id, p.sharpness, {expr} AS g FROM photos p
                  WHERE p.media_type = 'image' AND NOT p.is_video_frame {where}) t
            WHERE g IS NOT NULL GROUP BY g ORDER BY n DESC, g LIMIT %(top)s
            """,
            {**params, "top": args.top_n},
        ).fetchall()
        total = one(
            conn,
            f"SELECT count(*) AS n FROM photos p WHERE p.media_type = 'image' AND NOT p.is_video_frame {where}",
            params,
        )["n"]
    return {
        "group_by": args.group_by,
        "total_photos_matching_filters": total,
        "groups": [
            {"value": r["value"], "count": r["n"], "example_photo_id": str(r["example"])}
            for r in rows
        ],
    }


# ------------------------------------------------------------------ details


class DetailsArgs(BaseModel):
    photo_id: uuid.UUID


def get_photo_details(args: DetailsArgs) -> dict[str, Any]:
    with get_conn() as conn:
        r = conn.execute(
            "SELECT p.id, p.path, p.taken_at, p.place_name, p.admin1, p.country, p.camera, p.lens, "
            "p.focal_length, p.aperture, p.iso, p.shutter, p.width, p.height, p.keywords, "
            "(SELECT caption FROM captions c WHERE c.photo_id = p.id LIMIT 1) AS caption, "
            "(SELECT text FROM ocr_text o WHERE o.photo_id = p.id) AS ocr_text, "
            "(SELECT array_agg(DISTINCT pe.name) FROM faces f JOIN people pe ON pe.cluster_id = f.cluster_id "
            " WHERE f.photo_id = p.id AND pe.name IS NOT NULL) AS people "
            "FROM photos p WHERE p.id = %s",
            (args.photo_id,),
        ).fetchone()
    if r is None:
        return {"error": "no photo with that id"}
    out = {
        k: (str(v) if isinstance(v, (date, uuid.UUID)) else v)
        for k, v in r.items()
        if v not in (None, [], "")
    }
    out["photo_id"] = str(r["id"])
    out.pop("id", None)
    return out


# ------------------------------------------------------------------ registry

FILTERS_DOC = (
    "filters fields (all optional): date_from, date_to (YYYY-MM-DD, inclusive), months [1-12], years, "
    "place (city/region/country), camera, lens (e.g. '50mm' or a brand), focal_min, focal_max, "
    "aperture_min, aperture_max, iso_min, iso_max, people [names], media ('photo'|'video'), album_id."
)


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    s = model.model_json_schema()
    s.pop("title", None)
    return s


TOOLS: dict[str, tuple[type[BaseModel], Any, str]] = {
    "semantic_search": (
        SemanticSearchArgs, semantic_search,
        "Find photos by visual content, optionally within filters. sort='newest' returns the `limit` most "
        "relevant photos ordered newest first: check the captions to see which actually match, then "
        "the first matching one answers 'when did I last...'. " + FILTERS_DOC,
    ),
    "metadata_query": (
        MetadataQueryArgs, metadata_query,
        "Run a named, parameterised metadata query. Templates: "
        + "; ".join(f"{k}: {v[2]} params={list(v[0].model_fields)}" for k, v in TEMPLATES.items()),
    ),
    "aggregate": (
        AggregateArgs, aggregate,
        "Count photos grouped by city/region/country/camera/lens/year/month/weekday/hour/focal_length/aperture/person, "
        "within filters. Use for 'most photographed', 'how many', 'which camera'. " + FILTERS_DOC,
    ),
    "get_photo_details": (DetailsArgs, get_photo_details, "Full metadata, caption, OCR text and people for one photo id."),
}  # fmt: skip


def tool_specs() -> list[dict[str, Any]]:
    return [
        {"name": n, "description": d, "parameters": _schema(m)} for n, (m, _, d) in TOOLS.items()
    ]


@traceable(run_type="tool", name="run_tool")
def run_tool(name: str, raw_args: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Validate + execute. Returns (result, photo ids mentioned in the result).
    Errors come back as {"error": ...} so the model can correct itself."""
    if name not in TOOLS:
        return {"error": f"unknown tool {name!r}; available: {sorted(TOOLS)}"}, set()
    model, fn, _ = TOOLS[name]
    try:
        args = model(**(raw_args or {}))
        result = fn(args)
    except ValidationError as e:
        return {
            "error": "invalid arguments",
            "details": json.loads(e.json(include_url=False))[:5],
        }, set()
    except Exception as e:  # a tool failure must never crash the agent
        return {"error": f"{type(e).__name__}: {e}"}, set()
    return result, _ids_in(result)


def _ids_in(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("photo_id", "example_photo_id", "cover_photo_id") and isinstance(v, str):
                found.add(v)
            else:
                found |= _ids_in(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _ids_in(v)
    return found
