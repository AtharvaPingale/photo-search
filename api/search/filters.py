"""Structured filters and their translation to parameterised SQL.

Every value reaches Postgres as a bound parameter; the only strings spliced
into SQL are fixed fragments from this file. The query parser, the UI filter
chips and the agent's tools all produce a `SearchFilters`, so there is exactly
one place that turns user-influenced input into WHERE clauses.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Near(BaseModel):
    lat: float
    lon: float
    radius_km: float = 25.0


class SearchFilters(BaseModel):
    date_from: date | None = Field(None, description="inclusive, wall-clock date")
    date_to: date | None = Field(None, description="inclusive, wall-clock date")
    months: list[int] | None = Field(None, description="1-12, any year (e.g. 'summer photos')")
    years: list[int] | None = None
    place: str | None = Field(None, description="city, state/region or country name")
    near: Near | None = None
    camera: str | None = None
    lens: str | None = None
    focal_min: float | None = None
    focal_max: float | None = None
    aperture_min: float | None = None
    aperture_max: float | None = None
    iso_min: int | None = None
    iso_max: int | None = None
    people: list[str] = Field(default_factory=list, description="all must appear")
    media: Literal["photo", "video"] | None = None
    orientation: Literal["landscape", "portrait", "square"] | None = None
    keywords: list[str] = Field(default_factory=list)
    album_id: str | None = None

    @field_validator("months")
    @classmethod
    def _months(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return None
        v = sorted({m for m in v if 1 <= m <= 12})
        return v or None

    def is_empty(self) -> bool:
        return not any(
            v not in (None, [], "") for v in self.model_dump(exclude_none=False).values()
        )

    def chips(self) -> list[dict[str, Any]]:
        """Human-readable representation for the UI's editable filter chips."""
        out = []
        for k, v in self.model_dump(exclude_none=True).items():
            if v in ([], ""):
                continue
            out.append({"field": k, "value": v})
        return out


def to_sql(f: SearchFilters | None, alias: str = "p") -> tuple[str, dict[str, Any]]:
    """(sql, params). sql is '' or starts with ' AND '. Param names are prefixed f_."""
    if f is None:
        return "", {}
    a = alias
    parts: list[str] = []
    params: dict[str, Any] = {}

    if f.date_from:
        parts.append(f"{a}.taken_at >= %(f_date_from)s")
        params["f_date_from"] = f.date_from
    if f.date_to:
        parts.append(f"{a}.taken_at < %(f_date_to)s")
        params["f_date_to"] = f.date_to + timedelta(days=1)
    if f.months:
        parts.append(f"EXTRACT(MONTH FROM {a}.taken_at)::int = ANY(%(f_months)s)")
        params["f_months"] = f.months
    if f.years:
        parts.append(f"EXTRACT(YEAR FROM {a}.taken_at)::int = ANY(%(f_years)s)")
        params["f_years"] = f.years
    if f.place:
        parts.append(
            f"({a}.place_name ILIKE %(f_place)s OR {a}.admin1 ILIKE %(f_place)s "
            f"OR {a}.country ILIKE %(f_place)s)"
        )
        params["f_place"] = _escape_like(f.place.strip())
    if f.near:
        # equirectangular approximation: plenty accurate for tens of km
        parts.append(
            f"({a}.lat IS NOT NULL AND "
            f"sqrt(power(({a}.lat - %(f_near_lat)s) * 111.32, 2) + "
            f"power(({a}.lon - %(f_near_lon)s) * 111.32 * cos(radians(%(f_near_lat)s)), 2))"
            f" <= %(f_near_r)s)"
        )
        params.update(f_near_lat=f.near.lat, f_near_lon=f.near.lon, f_near_r=f.near.radius_km)
    if f.camera:
        parts.append(f"{a}.camera ILIKE %(f_camera)s")
        params["f_camera"] = f"%{_escape_like(f.camera.strip())}%"
    if f.lens:
        mm = re.fullmatch(r"(\d{1,3}(?:\.\d)?)\s?mm", f.lens.strip(), re.I)
        if mm:
            # "50mm" must match "FE 50mm F1.8" and "EF50mm" but not "18-150mm" or "150mm"
            parts.append(f"{a}.lens ~* %(f_lens)s")
            params["f_lens"] = rf"(^|[^0-9.\-]){re.escape(mm[1])}(\.0)?\s?mm"
        else:
            parts.append(f"replace({a}.lens, ' ', '') ILIKE %(f_lens)s")
            params["f_lens"] = f"%{_escape_like(f.lens.replace(' ', ''))}%"
    for fld, col, op in (
        ("focal_min", "focal_length", ">="),
        ("focal_max", "focal_length", "<="),
        ("aperture_min", "aperture", ">="),
        ("aperture_max", "aperture", "<="),
        ("iso_min", "iso", ">="),
        ("iso_max", "iso", "<="),
    ):
        val = getattr(f, fld)
        if val is not None:
            parts.append(f"{a}.{col} {op} %(f_{fld})s")
            params[f"f_{fld}"] = val
    for i, person in enumerate(f.people):
        parts.append(
            f"EXISTS (SELECT 1 FROM faces ff JOIN people pe ON pe.cluster_id = ff.cluster_id "
            f"WHERE ff.photo_id = {a}.id AND (lower(pe.name) = lower(%(f_person{i})s) "
            f"OR EXISTS (SELECT 1 FROM unnest(pe.aliases) al WHERE lower(al) = lower(%(f_person{i})s))))"
        )
        params[f"f_person{i}"] = person.strip()
    if f.media == "video":
        parts.append(f"{a}.is_video_frame")
    elif f.media == "photo":
        parts.append(f"NOT {a}.is_video_frame")
    if f.orientation == "landscape":
        parts.append(f"{a}.width > {a}.height * 1.05")
    elif f.orientation == "portrait":
        parts.append(f"{a}.height > {a}.width * 1.05")
    elif f.orientation == "square":
        parts.append(f"abs({a}.width - {a}.height) <= 0.05 * greatest({a}.width, {a}.height)")
    if f.keywords:
        parts.append(
            f"EXISTS (SELECT 1 FROM unnest({a}.keywords) kw WHERE lower(kw) = ANY(%(f_keywords)s))"
        )
        params["f_keywords"] = [k.lower() for k in f.keywords]
    if f.album_id:
        parts.append(
            f"EXISTS (SELECT 1 FROM album_photos ap WHERE ap.photo_id = {a}.id "
            f"AND ap.album_id = %(f_album)s::uuid)"
        )
        params["f_album"] = f.album_id

    sql = "".join(f" AND {p}" for p in parts)
    return sql, params


def lens_matches(filter_value: str, lens: str) -> bool:
    """Python mirror of the SQL lens filter, so the parser can check a proposed
    lens filter would match something in the library."""
    mm = re.fullmatch(r"(\d{1,3}(?:\.\d)?)\s?mm", filter_value.strip(), re.I)
    if mm:
        return re.search(rf"(^|[^0-9.\-]){re.escape(mm[1])}(\.0)?\s?mm", lens, re.I) is not None
    return filter_value.replace(" ", "").lower() in lens.replace(" ", "").lower()


def _escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
