"""Natural-language query -> (semantic text, structured filters).

"street shots at night with the 35mm in Tokyo last summer" becomes
    semantic = "street shots at night"
    filters  = lens 35mm, place Tokyo, dates 2026-06-01..2026-08-31

Two parsers share one output type:
  * RuleParser: deterministic, ~0.1 ms, driven by the library's own vocabulary
    (cameras, lenses, places, people actually present in the database).
  * LLM parser: a strict JSON schema, today's date and the same vocabulary in
    the prompt, then validated against that vocabulary.

mode="auto" (the default) sends only queries with metadata cues to the LLM:
"dog on a beach" never pays for an LLM round-trip, and any LLM failure or
timeout falls back to the rule result.
"""

from __future__ import annotations

import calendar
import difflib
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from api.search.filters import SearchFilters, lens_matches

Mode = Literal["auto", "hybrid", "llm", "llm_raw", "rules", "off"]


class ParsedQuery(BaseModel):
    text: str
    semantic: str
    filters: SearchFilters
    source: Literal["llm", "rules", "none", "user"] = "none"
    error: str | None = None
    latency_ms: float = 0.0


# --------------------------------------------------------------- vocabulary


@dataclass
class Vocab:
    cameras: list[str] = field(default_factory=list)
    lenses: list[str] = field(default_factory=list)
    places: list[str] = field(
        default_factory=list
    )  # place_name, admin1, country; most frequent first
    people: list[str] = field(default_factory=list)  # names and aliases


_vocab_cache: tuple[float, Vocab] | None = None
_vocab_lock = threading.Lock()
VOCAB_TTL_S = 60.0


def load_vocab(conn=None, *, fresh: bool = False) -> Vocab:
    global _vocab_cache
    with _vocab_lock:
        if not fresh and _vocab_cache and time.monotonic() - _vocab_cache[0] < VOCAB_TTL_S:
            return _vocab_cache[1]
    from api.db.session import get_conn

    def _q(c) -> Vocab:
        def col(sql: str) -> list[str]:
            return [r["v"] for r in c.execute(sql).fetchall() if r["v"]]

        places = col(
            "SELECT v FROM (SELECT place_name v, count(*) n FROM photos GROUP BY 1 UNION ALL "
            "SELECT admin1, count(*) FROM photos GROUP BY 1 UNION ALL "
            "SELECT country, count(*) FROM photos GROUP BY 1) t WHERE v IS NOT NULL "
            "GROUP BY v ORDER BY sum(n) DESC LIMIT 2000"
        )
        people = col("SELECT name v FROM people WHERE name IS NOT NULL AND NOT hidden")
        people += col("SELECT DISTINCT unnest(aliases) v FROM people WHERE NOT hidden")
        return Vocab(
            cameras=col("SELECT camera v FROM photos GROUP BY 1 ORDER BY count(*) DESC LIMIT 200"),
            lenses=col("SELECT lens v FROM photos GROUP BY 1 ORDER BY count(*) DESC LIMIT 200"),
            places=places,
            people=sorted(set(people), key=str.lower),
        )

    if conn is not None:
        v = _q(conn)
    else:
        with get_conn() as c:
            v = _q(c)
    with _vocab_lock:
        _vocab_cache = (time.monotonic(), v)
    return v


def clear_vocab_cache() -> None:
    global _vocab_cache
    with _vocab_lock:
        _vocab_cache = None


# --------------------------------------------------------------- date helpers

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9
AMBIGUOUS_MONTHS = {"may", "march", "mar", "jan", "sat", "jun"}  # also ordinary words
SEASONS = {
    "spring": (3, 5),
    "summer": (6, 8),
    "fall": (9, 11),
    "autumn": (9, 11),
    "winter": (12, 2),
}


def month_range(y: int, m: int) -> tuple[date, date]:
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def season_range(y: int, season: str) -> tuple[date, date]:
    """Winter Y = Dec Y through Feb Y+1."""
    a, b = SEASONS[season]
    if a > b:
        return date(y, a, 1), month_range(y + 1, b)[1]
    return date(y, a, 1), month_range(y, b)[1]


def last_season(today: date, season: str) -> tuple[date, date]:
    """Most recent *completed* season. In September, 'last summer' is this year's."""
    for y in (today.year, today.year - 1, today.year - 2):
        start, end = season_range(y, season)
        if end < today:
            return start, end
    return season_range(today.year - 1, season)


def this_season(today: date, season: str) -> tuple[date, date]:
    start, end = season_range(today.year, season)
    if season == "winter" and today.month <= 2:
        start, end = season_range(today.year - 1, season)
    return start, end


def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


# --------------------------------------------------------------- rule parser

YEAR = r"(?:19[5-9]\d|20\d\d)"
_FILLER = re.compile(
    r"\b(?:photos?|pictures?|pics?|images?|shots? of|snaps?|show me|find|search for|"
    r"all|my|any|some|taken|shot|captured)\b",
    re.I,
)
_DANGLING = re.compile(r"^(?:(?:of|from|in|on|at|with|the|and|during|using|near|around)\s+)+|"
                       r"(?:\s+(?:of|from|in|on|at|with|the|and|during|using|near|around))+$", re.I)  # fmt: skip
_BRANDS = ["canon", "nikon", "sony", "fujifilm", "fuji", "leica", "ricoh", "olympus", "panasonic",
           "lumix", "pentax", "hasselblad", "iphone", "pixel", "galaxy", "gopro", "dji"]  # fmt: skip
# GeoNames has towns called all of these; in a query they're almost always just words.
COMMON_WORD_PLACES = {
    "golden", "beach", "sunset", "sunrise", "bath", "reading", "nice", "mobile", "orange",
    "paradise", "eagle", "hope", "harmony", "liberty", "victoria", "florence", "garden",
    "mountain", "lake", "river", "park", "forest", "valley", "bay", "harbor", "harbour",
    "ocean", "surf", "snow", "rain", "sun", "moon", "star", "rose", "lily", "ivy", "dog",
    "cat", "bear", "wolf", "fox", "deer", "bird", "horse", "salt", "sand", "rock", "stone",
    "hill", "hills", "field", "fields", "spring", "summer", "winter", "fall", "autumn",
    "day", "night", "home", "street", "bridge", "castle", "church", "tower", "market",
    "canyon", "desert", "island", "coast", "view", "vista", "grand", "diamond", "pearl",
    "crystal", "silver", "gold", "red", "green", "white", "black", "blue", "brown",
    "friendship", "unity", "concord", "independence", "enterprise", "commerce", "industry",
    "fairview", "plain", "plains", "lincoln", "jackson",
    "ali", "bar", "man", "most", "split", "cork", "kiss", "mary", "jordan", "chad", "georgia",
}  # fmt: skip
_LENS_BRANDS = ["sigma", "tamron", "zeiss", "samyang", "rokinon", "viltrox", "voigtlander",
                "voigtländer", "ttartisans", "7artisans", "laowa", "tokina", "helios"]  # fmt: skip
_CUES = re.compile(
    rf"\b(?:{YEAR}|today|yesterday|week|weekend|month|year|ago|last|this|past|since|before|after|"
    r"during|spring|summer|fall|autumn|winter|christmas|halloween|thanksgiving|birthday|"
    r"iso|shot on|mm|f/?\d|video|clip|footage|vertical|horizontal|"
    + "|".join(k for k in MONTHS if len(k) > 3)
    + r")\b|\d+\s?mm|\bf/?\d",
    re.I,
)


@dataclass
class _Ctx:
    text: str
    filters: dict[str, Any] = field(default_factory=dict)

    def consume(self, m: re.Match[str]) -> None:
        a, b = m.span()
        self.text = self.text[:a] + " " * (b - a) + self.text[b:]


def _set_range(ctx: _Ctx, start: date, end: date) -> None:
    ctx.filters["date_from"] = start
    ctx.filters["date_to"] = end


def _parse_dates(ctx: _Ctx, today: date) -> None:
    t = ctx.text
    pre = r"(?:\b(?:in|from|during|on|of)\s+)?"

    def sub(pattern: str, fn) -> None:
        nonlocal t
        m = re.search(pattern, ctx.text, re.I)
        if m and fn(m) is not False:
            ctx.consume(m)
        t = ctx.text

    # explicit year ranges
    sub(rf"{pre}(?:between\s+)?({YEAR})\s*(?:-|–|to|and|through)\s*({YEAR})\b",
        lambda m: _set_range(ctx, date(int(m[1]), 1, 1), date(int(m[2]), 12, 31)))  # fmt: skip
    sub(rf"\bsince\s+({YEAR})\b", lambda m: _set_range(ctx, date(int(m[1]), 1, 1), today))
    sub(
        rf"\bbefore\s+({YEAR})\b",
        lambda m: ctx.filters.__setitem__("date_to", date(int(m[1]) - 1, 12, 31)),
    )

    # holidays, only with a year or last/this ("thanksgiving dinner last year" too)
    sub(rf"\b(?:(last|this)\s+)?(?:({YEAR})\s+)?(christmas|halloween|new year'?s eve|thanksgiving)"
        rf"(?:\s+({YEAR}))?(?:(?:\s+\w+)?\s+(last|this)\s+year)?\b",
        lambda m: _holiday_match(ctx, m, today))  # fmt: skip

    # relative
    rel = {
        "today": (today, today),
        "yesterday": (today - timedelta(days=1), today - timedelta(days=1)),
        "this week": (_week_start(today), today),
        "last week": (
            _week_start(today) - timedelta(days=7),
            _week_start(today) - timedelta(days=1),
        ),
        "this month": (today.replace(day=1), today),
        "last month": month_range(
            *((today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12))
        ),
        "this year": (date(today.year, 1, 1), today),
        "last year": (date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)),
        "last weekend": (
            _week_start(today) - timedelta(days=2),
            _week_start(today) - timedelta(days=1),
        ),
    }
    for phrase, (a, b) in rel.items():
        sub(rf"{pre}\b{phrase}\b", lambda m, a=a, b=b: _set_range(ctx, a, b))
    sub(r"\b(?:in\s+)?(?:the\s+)?(?:last|past)\s+(\d+|a|one|two|three|six)\s+(day|week|month|year)s?\b",
        lambda m: _set_range(ctx, today - _span(m[1], m[2]), today))  # fmt: skip

    # seasons
    seasons = "spring|summer|fall|autumn|winter"
    sub(rf"{pre}\b(last|this)\s+({seasons})\b",
        lambda m: _set_range(ctx, *(last_season if m[1].lower() == "last" else this_season)(today, m[2].lower())))  # fmt: skip
    sub(rf"{pre}\b({seasons})\s+(?:of\s+)?({YEAR})\b",
        lambda m: _set_range(ctx, *season_range(int(m[2]), m[1].lower())))  # fmt: skip
    sub(rf"\b(?:in|from|during)\s+(?:the\s+)?({seasons})\b",
        lambda m: ctx.filters.__setitem__("months", _season_months(m[1].lower())))  # fmt: skip

    # months
    month_names = "|".join(sorted(MONTHS, key=len, reverse=True))
    sub(rf"{pre}\b({month_names})\.?\s+(?:of\s+)?({YEAR})\b",
        lambda m: _set_range(ctx, *month_range(int(m[2]), MONTHS[m[1].lower()])))  # fmt: skip
    sub(rf"\b(last|this)\s+({month_names})\b", lambda m: _last_month_named(ctx, m, today))

    def bare_month(m: re.Match[str]) -> bool | None:
        word = m[2]
        if word.lower() in AMBIGUOUS_MONTHS and not (m[1] or word[0].isupper()):
            return False
        ctx.filters.setdefault("months", [])
        ctx.filters["months"] = sorted({*ctx.filters["months"], MONTHS[word.lower()]})
        return None

    sub(rf"(\b(?:in|from|during)\s+)?\b({month_names})\b(?!\s*\d)", bare_month)

    # a bare year: "tokyo 2019", "in 2024"
    def year(m: re.Match[str]) -> None:
        y = int(m[1])
        if "date_from" in ctx.filters and ctx.filters["date_from"].year != y:
            return
        if ctx.filters.get("months"):
            ctx.filters.setdefault("years", []).append(y)
        else:
            _set_range(ctx, date(y, 1, 1), date(y, 12, 31))

    sub(rf"{pre}\b({YEAR})\b", year)


def _holiday_match(ctx: _Ctx, m: re.Match[str], today: date) -> bool | None:
    rel, y1, name, y2, rel_year = m[1], m[2], m[3].lower(), m[4], m[5]
    if y1 or y2:
        y = int(y1 or y2)
    elif rel_year:
        y = today.year - 1 if rel_year.lower() == "last" else today.year
        # keep the word between the holiday and "last year" ("dinner") in the semantic text
        between = m[0][m.end(3) - m.start(0) : m.start(5) - m.start(0)]
        between = re.sub(r"\b(last|this)\s*$", "", between).strip()
        _set_range(ctx, *_holiday(y, name))
        if between:
            ctx.text = ctx.text[: m.start(0)] + " " + between + " " + ctx.text[m.end(0) :]
            return False
        return None
    elif rel:
        y = today.year if rel.lower() == "this" else today.year - 1
        a, _ = _holiday(today.year, name)
        if rel.lower() == "last" and a < today:
            y = today.year
    else:
        return False  # bare "christmas" stays semantic (christmas tree, lights...)
    _set_range(ctx, *_holiday(y, name))
    return None


def _holiday(y: int, name: str) -> tuple[date, date]:
    if name.startswith("christmas"):
        return date(y, 12, 24), date(y, 12, 26)
    if name.startswith("halloween"):
        return date(y, 10, 31), date(y, 10, 31)
    if name.startswith("new year"):
        return date(y, 12, 31), date(y + 1, 1, 1)
    # thanksgiving: 4th Thursday of November
    first = date(y, 11, 1)
    thu = first + timedelta(days=(3 - first.weekday()) % 7 + 21)
    return thu, thu + timedelta(days=3)


def _last_month_named(ctx: _Ctx, m: re.Match[str], today: date) -> None:
    mo = MONTHS[m[2].lower()]
    y = today.year
    if m[1].lower() == "last" and (mo >= today.month):
        y -= 1
    _set_range(ctx, *month_range(y, mo))


def _season_months(season: str) -> list[int]:
    a, b = SEASONS[season]
    return [12, 1, 2] if a > b else list(range(a, b + 1))


_NUM = {"a": 1, "one": 1, "two": 2, "three": 3, "six": 6}


def _span(n: str, unit: str) -> timedelta:
    k = _NUM.get(n.lower()) or int(n)
    days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit.lower()]
    return timedelta(days=k * days)


def _is_prime_named(lens: str, mm: str) -> bool:
    return re.search(rf"(?<![\d.\-]){re.escape(mm)}(?:\.0)?\s?mm", lens, re.I) is not None


def _camera_aliases(camera: str) -> list[str]:
    out = {camera}
    parts = camera.split()
    if len(parts) > 1:
        out.add(" ".join(parts[1:]))
    for p in parts:
        if re.search(r"\d", p) and len(p) >= 2 and not re.fullmatch(r"\d+", p):
            out.add(p)
    return sorted(out, key=len, reverse=True)


class RuleParser:
    def parse(self, q: str, vocab: Vocab, today: date | None = None) -> ParsedQuery:
        t0 = time.perf_counter()
        today = today or date.today()
        ctx = _Ctx(text=q)

        _parse_dates(ctx, today)
        self._equipment(ctx, vocab)
        self._people(ctx, vocab)
        self._places(ctx, vocab)
        self._media(ctx)

        semantic = _clean(ctx.text)
        try:
            filters = SearchFilters(**ctx.filters)
        except ValidationError as e:
            return ParsedQuery(
                text=q, semantic=q, filters=SearchFilters(), source="rules", error=str(e)
            )
        return ParsedQuery(
            text=q,
            semantic=semantic,
            filters=filters,
            source="rules",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def _equipment(self, ctx: _Ctx, vocab: Vocab) -> None:
        # zoom by its range: "with the 24-70", "the 100-400mm"
        m = re.search(
            r"\b(?:with|on|using)?\s*(?:the|my)?\s*(\d{2,3})\s?-\s?(\d{2,3})\s?(?:mm)?\b(?:\s+lens)?",
            ctx.text,
            re.I,
        )
        if m and any(re.search(rf"{m[1]}\s?-\s?{m[2]}", lens) for lens in vocab.lenses):
            ctx.filters["lens"] = f"{m[1]}-{m[2]}"
            ctx.consume(m)
        # focal bounds: "under 24mm", "over 200mm"
        m = re.search(
            r"\b(under|below|less than|over|above|beyond)\s+(\d{1,3})\s?mm\b", ctx.text, re.I
        )
        if m:
            key = "focal_max" if m[1].lower() in ("under", "below", "less than") else "focal_min"
            ctx.filters[key] = float(m[2])
            ctx.consume(m)
        # focal length: "the 50mm", "85 mm"
        m = re.search(
            r"\b(?:with|on|using|at)?\s*(?:the|my|a)?\s*(\d{1,3}(?:\.\d)?)\s?mm\b(?:\s+lens)?",
            ctx.text,
            re.I,
        )
        if m:
            mm = m[1]
            primes = [lens for lens in vocab.lenses if _is_prime_named(lens, mm)]
            if primes:
                ctx.filters["lens"] = f"{mm}mm"
            else:
                f = float(mm)
                ctx.filters["focal_min"], ctx.filters["focal_max"] = (
                    round(f * 0.95, 1),
                    round(f * 1.05, 1),
                )
            ctx.consume(m)
        # lens brand
        lens_brands = {b for lens in vocab.lenses for b in _LENS_BRANDS if b in lens.lower()}
        for b in sorted(lens_brands, key=len, reverse=True):
            m = re.search(
                rf"\b(?:with|on|using)?\s*(?:the|my)?\s*{re.escape(b)}\b(?:\s+lens)?",
                ctx.text,
                re.I,
            )
            if m:
                ctx.filters["lens"] = b
                ctx.consume(m)
                break
        # aperture / iso
        m = re.search(r"\b(?:at\s+)?f\s?/?\s?(\d{1,2}(?:\.\d)?)\b", ctx.text, re.I)
        if m and float(m[1]) <= 32:
            v = float(m[1])
            ctx.filters["aperture_min"], ctx.filters["aperture_max"] = v - 0.05, v + 0.05
            ctx.consume(m)
        m = re.search(r"\b(?:at\s+)?iso\s?(\d{2,6})\b", ctx.text, re.I)
        if m:
            ctx.filters["iso_min"] = ctx.filters["iso_max"] = int(m[1])
            ctx.consume(m)
        m = re.search(r"\bhigh[\s-]iso\b", ctx.text, re.I)
        if m:
            ctx.filters["iso_min"] = 3200
            ctx.consume(m)
        # camera: exact vocabulary aliases first, then brand names
        for cam in vocab.cameras:
            for alias in _camera_aliases(cam):
                m = re.search(
                    rf"\b(?:with|on|using|shot on)?\s*(?:the|my|an?)?\s*{re.escape(alias)}\b",
                    ctx.text,
                    re.I,
                )
                if m:
                    ctx.filters["camera"] = alias
                    ctx.consume(m)
                    return
        cam_text = " ".join(vocab.cameras).lower()
        for word, brand in [(b, b) for b in _BRANDS] + [("drone", "dji")]:
            if brand in cam_text:
                m = re.search(
                    rf"\b(?:with|on|using|shot on|from)?\s*(?:the|my|an?)?\s*{word}\b",
                    ctx.text,
                    re.I,
                )
                if m:
                    ctx.filters["camera"] = "fuji" if brand == "fujifilm" else brand
                    ctx.consume(m)
                    return

    def _people(self, ctx: _Ctx, vocab: Vocab) -> None:
        found = []
        for name in sorted(vocab.people, key=len, reverse=True):
            m = re.search(rf"\b(?:with\s+)?{re.escape(name)}\b(?:\s+and\b)?", ctx.text, re.I)
            if m:
                found.append(name)
                ctx.consume(m)
        if found:
            ctx.filters["people"] = found

    def _places(self, ctx: _Ctx, vocab: Vocab) -> None:
        for place in sorted(vocab.places, key=len, reverse=True):
            # Place names that are also ordinary words ("golden hour" vs Golden, CO) only
            # count with a preposition ("in golden") or when the user capitalised them.
            m = re.search(
                rf"\b(in|at|from|near|around|to)\s+(?:the\s+)?({re.escape(place)})\b",
                ctx.text,
                re.I,
            )
            if not m:
                m2 = re.search(rf"\b({re.escape(place)})\b", ctx.text, re.I)
                if not m2:
                    continue
                if place.lower() in COMMON_WORD_PLACES and not m2[1][0].isupper():
                    continue
                m = m2
            ctx.filters["place"] = place
            ctx.consume(m)
            return

    def _media(self, ctx: _Ctx) -> None:
        m = re.search(r"\b(?:videos?|clips?|footage)\b", ctx.text, re.I)
        if m:
            ctx.filters["media"] = "video"
            ctx.consume(m)
        m = re.search(
            r"\b(?:in\s+)?(vertical|portrait orientation|horizontal|landscape orientation)\b",
            ctx.text,
            re.I,
        )
        if m:
            ctx.filters["orientation"] = (
                "portrait" if m[1].lower().startswith(("vert", "portrait")) else "landscape"
            )
            ctx.consume(m)


def _clean(text: str) -> str:
    t = _FILLER.sub(" ", text)
    t = re.sub(r"(^|\s)'s\b", " ", t)  # "Mom's birthday" minus "Mom" leaves "'s"
    t = re.sub(r"\s+", " ", t).strip(" ,.;:-")
    prev = None
    while prev != t:
        prev = t
        t = _DANGLING.sub("", t).strip(" ,.;:-")
    return t


def has_metadata_cues(q: str, vocab: Vocab) -> bool:
    if _CUES.search(q):
        return True
    low = q.lower()
    if any(p.lower() in low for p in vocab.people):
        return True
    # a capitalised word that isn't sentence-initial is often a place or name
    words = q.split()
    return any(w[:1].isupper() for w in words[1:])


# --------------------------------------------------------------- LLM parser

_NULLABLE = lambda t: {"anyOf": [{"type": t}, {"type": "null"}]}  # noqa: E731

LLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "semantic_query": {"type": "string"},
        "date_from": _NULLABLE("string"),
        "date_to": _NULLABLE("string"),
        "months": {"anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "null"}]},
        "place": _NULLABLE("string"),
        "camera": _NULLABLE("string"),
        "lens": _NULLABLE("string"),
        "focal_min": _NULLABLE("number"),
        "focal_max": _NULLABLE("number"),
        "aperture_min": _NULLABLE("number"),
        "aperture_max": _NULLABLE("number"),
        "iso_min": _NULLABLE("integer"),
        "iso_max": _NULLABLE("integer"),
        "people": {"type": "array", "items": {"type": "string"}},
        "media": {"anyOf": [{"type": "string", "enum": ["photo", "video"]}, {"type": "null"}]},
        "orientation": {
            "anyOf": [
                {"type": "string", "enum": ["landscape", "portrait", "square"]},
                {"type": "null"},
            ]
        },
    },
    "required": [
        "semantic_query",
        "date_from",
        "date_to",
        "months",
        "place",
        "camera",
        "lens",
        "focal_min",
        "focal_max",
        "aperture_min",
        "aperture_max",
        "iso_min",
        "iso_max",
        "people",
        "media",
        "orientation",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You convert a photo-library search query into structured filters.

Return JSON matching the schema. Rules:
- semantic_query: the visual content only (subjects, scenes, mood, lighting, colours, actions).
  Remove anything you turned into a filter. Keep visual phrases such as "golden hour",
  "at night", "sunset", "portrait" (the genre), "landscape" (the genre). Empty string if nothing visual remains.
- Dates are wall-clock dates, YYYY-MM-DD, inclusive. Resolve relative dates against TODAY.
  Seasons are northern-hemisphere (summer = Jun-Aug; winter Y = Dec Y to Feb Y+1).
  "last summer" is the most recent summer that has already ended.
  A month or season without a year ("photos from June") -> months, not dates.
- place: only a city, region or country. Prefer spellings from KNOWN PLACES.
- people: only names from KNOWN PEOPLE (including aliases such as "me"). Never invent names.
- camera / lens: short substrings that match KNOWN CAMERAS / KNOWN LENSES.
  "the 50mm" -> lens "50mm" when a known prime lens has that focal length; otherwise
  focal_min/focal_max within 5% of the number.
- "f/1.8" -> aperture_min 1.75, aperture_max 1.85. "ISO 3200" -> iso_min = iso_max = 3200.
- media "video" only when the user asks for videos or clips.
- Use null / [] for anything not mentioned. Do not guess."""


def _llm_prompt(q: str, vocab: Vocab, today: date) -> str:
    def lst(xs: list[str], n: int) -> str:
        return ", ".join(xs[:n]) if xs else "(none)"

    return (
        f"TODAY: {today.isoformat()} ({today.strftime('%A')})\n"
        f"KNOWN CAMERAS: {lst(vocab.cameras, 40)}\n"
        f"KNOWN LENSES: {lst(vocab.lenses, 40)}\n"
        f"KNOWN PLACES: {lst(vocab.places, 150)}\n"
        f"KNOWN PEOPLE: {lst(vocab.people, 100)}\n\n"
        f"QUERY: {q}"
    )


def _match_vocab(value: str | None, options: list[str], cutoff: float = 0.85) -> str | None:
    if not value:
        return None
    low = {o.lower(): o for o in options}
    if value.lower() in low:
        return low[value.lower()]
    close = difflib.get_close_matches(value.lower(), list(low), n=1, cutoff=cutoff)
    return low[close[0]] if close else value


def llm_parse(
    q: str,
    vocab: Vocab,
    today: date | None = None,
    timeout: float | None = None,
    *,
    validate: bool = True,
) -> ParsedQuery:
    from api.llm import complete_json

    t0 = time.perf_counter()
    today = today or date.today()
    raw = complete_json(SYSTEM_PROMPT, _llm_prompt(q, vocab, today), LLM_SCHEMA, timeout=timeout)
    semantic = (raw.get("semantic_query") or "").strip()
    f = validate_llm_fields(raw, q, vocab) if validate else _coerce(raw, vocab)
    if validate:
        # a value we refused as a filter goes back into the semantic text
        for v in _dropped_text(raw, f, q):
            if v.lower() not in semantic.lower():
                semantic = f"{semantic} {v}".strip()
    try:
        filters = SearchFilters(**{k: v for k, v in f.items() if v not in (None, [])})
    except ValidationError as e:
        return ParsedQuery(text=q, semantic=q, filters=SearchFilters(), source="llm", error=str(e))
    return ParsedQuery(
        text=q,
        semantic=semantic,
        filters=filters,
        source="llm",
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


def _coerce(raw: dict[str, Any], vocab: Vocab) -> dict[str, Any]:
    f: dict[str, Any] = {
        k: raw.get(k)
        for k in ("date_from", "date_to", "months", "place", "camera", "lens", "focal_min", "focal_max",
                  "aperture_min", "aperture_max", "iso_min", "iso_max", "media", "orientation")
    }  # fmt: skip
    people_low = {p.lower(): p for p in vocab.people}
    f["people"] = [
        people_low[p.lower()] for p in raw.get("people") or [] if p.lower() in people_low
    ]
    for k in ("date_from", "date_to"):
        if f[k]:
            try:
                f[k] = date.fromisoformat(str(f[k])[:10])
            except ValueError:
                f[k] = None
    return f


def _mentions(q: str, token: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(token.lower())}(?![a-z0-9])", q.lower()) is not None


def validate_llm_fields(raw: dict[str, Any], q: str, vocab: Vocab) -> dict[str, Any]:
    """The LLM proposes, the code disposes: every filter needs evidence in the query
    text and, where there's a vocabulary, a match in the library. Small local models
    otherwise add plausible-looking filters nobody asked for (media="photo" on every
    query, a camera guessed from a lens, "beach" as a place)."""
    f = _coerce(raw, vocab)
    low = q.lower()
    if f["media"] != "video" or not re.search(r"\b(videos?|clips?|footage|movies?)\b", low):
        f["media"] = None
    if f["orientation"] and not re.search(
        r"\b(vertical|horizontal|portrait orientation|landscape orientation|square)\b", low
    ):
        f["orientation"] = None
    # place: must be a library place, mentioned in the query ("Tokyo, Japan" -> "Tokyo")
    place = None
    for part in [raw.get("place"), *str(raw.get("place") or "").split(",")]:
        m = _match_vocab((part or "").strip(), vocab.places, cutoff=0.9)
        if m and m in vocab.places and any(_mentions(q, w) for w in m.split() if len(w) > 2):
            place = m
            break
    f["place"] = place
    # camera: some distinctive token of it must appear in the query
    cam = f["camera"]
    if cam:
        toks = [t for t in re.findall(r"[a-z0-9]+", cam.lower()) if len(t) >= 2]
        brand_words = {"dji": ["drone"], "apple": ["iphone"], "fujifilm": ["fuji"]}
        ev = [t for t in toks if _mentions(q, t)] + [
            b
            for b, words in brand_words.items()
            if b in cam.lower() and any(_mentions(q, w) for w in words)
        ]
        f["camera"] = ev[0] if ev else None
    # lens: the focal length, zoom range or brand the user actually said
    lens = f["lens"]
    if lens:
        mm = re.search(r"(\d{2,3})\s?-\s?(\d{2,3})", q) or re.search(r"(\d{1,3})\s?mm", q, re.I)
        brands = [b for b in _LENS_BRANDS if b in lens.lower() and _mentions(q, b)]
        if mm and mm.lastindex == 2 and f"{mm[1]}-{mm[2]}" in lens.replace(" ", ""):
            f["lens"] = f"{mm[1]}-{mm[2]}"
        elif mm and mm.lastindex == 1 and mm[1] in lens:
            f["lens"] = f"{mm[1]}mm"
        elif brands:
            f["lens"] = brands[0]
        else:
            f["lens"] = None
    if f["lens"] and vocab.lenses and not any(lens_matches(f["lens"], v) for v in vocab.lenses):
        f["lens"] = None  # e.g. "400mm" when the only 400 is a 100-400mm zoom
    if f["lens"] and not re.search(r"\b(under|below|over|above|less than|beyond)\b", low):
        f["focal_min"] = f["focal_max"] = None  # the lens already pins focal length
    if not re.search(r"\d\s?mm|\bwide|\btele", low):
        f["focal_min"] = f["focal_max"] = None
    if not re.search(r"\bf\s?/?\s?\d", low):
        f["aperture_min"] = f["aperture_max"] = None
    if "iso" not in low:
        f["iso_min"] = f["iso_max"] = None
    f["people"] = [p for p in f["people"] if _mentions(q, p)]
    if not _DATE_CUES.search(q):
        f["date_from"] = f["date_to"] = f["months"] = None
    return f


_DATE_CUES = re.compile(
    rf"\b(?:{YEAR}|today|yesterday|week|weekend|month|year|ago|last|this|past|since|before|after|"
    r"spring|summer|fall|autumn|winter|christmas|halloween|thanksgiving|new year|"
    + "|".join(k for k in MONTHS if len(k) > 3)
    + r")\b",
    re.I,
)


def _dropped_text(raw: dict[str, Any], kept: dict[str, Any], q: str) -> list[str]:
    out = []
    for k in ("place", "camera", "lens"):
        v = raw.get(k)
        if v and not kept.get(k):
            for part in str(v).split(","):
                part = part.strip()
                if part and _mentions(q, part):
                    out.append(part)
    return out


DATE_FIELDS = ("date_from", "date_to", "months", "years")
FIELD_GROUPS = [
    DATE_FIELDS,
    ("focal_min", "focal_max"),
    ("aperture_min", "aperture_max"),
    ("iso_min", "iso_max"),
]


def merge(rules: ParsedQuery, llm: ParsedQuery) -> ParsedQuery:
    """Rules win wherever they found something (they're deterministic and high precision,
    and do date arithmetic better than a small LLM); the LLM fills in the rest."""
    r = rules.filters.model_dump()
    m = llm.filters.model_dump()
    out = dict(m)
    grouped = {f for g in FIELD_GROUPS for f in g}
    for group in FIELD_GROUPS:
        if any(r[k] not in (None, []) for k in group):
            for k in group:
                out[k] = r[k]
    for k, v in r.items():
        if k not in grouped and v not in (None, []):
            out[k] = v
    # the semantic text must not keep words the rules turned into filters
    semantic = llm.semantic
    if (
        rules.semantic
        and len(rules.semantic) < len(semantic)
        and set(rules.semantic.lower().split()) <= set(semantic.lower().split())
    ):
        semantic = rules.semantic
    return ParsedQuery(
        text=rules.text,
        semantic=semantic,
        filters=SearchFilters(**out),
        source="llm",
        latency_ms=rules.latency_ms + llm.latency_ms,
    )


# --------------------------------------------------------------- entry point

_cache: OrderedDict[tuple, ParsedQuery] = OrderedDict()
_cache_lock = threading.Lock()
_rules = RuleParser()


def parse_query(
    q: str,
    mode: Mode = "auto",
    *,
    vocab: Vocab | None = None,
    today: date | None = None,
) -> ParsedQuery:
    """mode: off | rules | llm (validated LLM alone) | llm_raw (unvalidated, for the eval)
    | hybrid (rules + validated LLM) | auto (hybrid only when metadata cues remain
    after the rules have taken what they understood)."""
    q = q.strip()
    today = today or date.today()
    if mode == "off" or not q:
        return ParsedQuery(text=q, semantic=q, filters=SearchFilters(), source="none")
    vocab = vocab if vocab is not None else load_vocab()
    rules = _rules.parse(q, vocab, today)
    if mode == "rules":
        return rules
    from api.llm import llm_enabled

    if not llm_enabled():
        return rules
    # auto: only what the rules couldn't explain goes to the LLM. "cats in Chicago"
    # is fully handled by the rules (place=Chicago, semantic "cats"), so it costs
    # 0.1 ms instead of a ~2 s LLM call; "the weekend before Rohan's wedding" isn't.
    if mode == "auto" and not has_metadata_cues(rules.semantic, vocab):
        return rules

    key = (q, mode, today, len(vocab.places), len(vocab.people), len(vocab.cameras))
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    try:
        llm = llm_parse(q, vocab, today, validate=mode != "llm_raw")
    except Exception as e:
        rules.error = f"llm parser failed, used rules: {e}"
        return rules
    parsed = llm if mode in ("llm", "llm_raw") else merge(rules, llm)
    with _cache_lock:
        _cache[key] = parsed
        while len(_cache) > 512:
            _cache.popitem(last=False)
    return parsed


def clear_parse_cache() -> None:
    with _cache_lock:
        _cache.clear()
