from __future__ import annotations

from datetime import date

import pytest

from api.search.query_parser import RuleParser, Vocab, last_season, season_range

TODAY = date(2026, 9, 26)
VOCAB = Vocab(
    cameras=["Canon EOS R6", "Fujifilm X100V", "Apple iPhone 15 Pro"],
    lenses=["RF50mm F1.8 STM", "FE 85mm F1.8", "RF24-70mm F2.8", "Sigma 35mm F1.4 DG"],
    places=["Chicago", "Illinois", "United States", "Tokyo", "Japan", "Golden"],
    people=["Rohan", "me"],
)
P = RuleParser()


def parse(q: str):
    return P.parse(q, VOCAB, TODAY)


def test_last_summer_is_most_recent_completed():
    assert last_season(TODAY, "summer") == (date(2026, 6, 1), date(2026, 8, 31))
    assert last_season(date(2026, 7, 1), "summer") == (date(2025, 6, 1), date(2025, 8, 31))
    assert season_range(2024, "winter") == (date(2024, 12, 1), date(2025, 2, 28))


@pytest.mark.parametrize(
    "q, semantic, expect",
    [
        ("dog on a beach", "dog on a beach", {}),
        ("golden hour on the beach", "golden hour on the beach", {}),
        (
            "photos from Chicago in 2025",
            "",
            {"place": "Chicago", "date_from": date(2025, 1, 1), "date_to": date(2025, 12, 31)},
        ),
        ("portraits with the 85mm", "portraits", {"lens": "85mm"}),
        ("street shots at night with the 35mm", "street shots at night", {"lens": "35mm"}),
        ("mountains with the 200mm", "mountains", {"focal_min": 190.0, "focal_max": 210.0}),
        (
            "sunset last summer",
            "sunset",
            {"date_from": date(2026, 6, 1), "date_to": date(2026, 8, 31)},
        ),
        (
            "snow in winter 2024",
            "snow",
            {"date_from": date(2024, 12, 1), "date_to": date(2025, 2, 28)},
        ),
        ("me and Rohan hiking", "hiking", {"people": ["Rohan", "me"]}),
        ("coffee shop menu", "coffee shop menu", {}),
        ("night street on the X100V", "night street", {"camera": "X100V"}),
        ("shot on iphone at the park", "park", {"camera": "iphone"}),
        ("tokyo videos", "", {"place": "Tokyo", "media": "video"}),
        (
            "fireworks June 2025",
            "fireworks",
            {"date_from": date(2025, 6, 1), "date_to": date(2025, 6, 30)},
        ),
        ("lake in June", "lake", {"months": [6]}),
        ("may flowers", "may flowers", {}),
        ("bokeh at f/1.8", "bokeh", {"aperture_min": 1.75, "aperture_max": 1.85}),
        ("concert iso 6400", "concert", {"iso_min": 6400, "iso_max": 6400}),
        (
            "beach 2019-2021",
            "beach",
            {"date_from": date(2019, 1, 1), "date_to": date(2021, 12, 31)},
        ),
        ("christmas tree", "christmas tree", {}),
        ("christmas 2024", "", {"date_from": date(2024, 12, 24), "date_to": date(2024, 12, 26)}),
        ("with the sigma", "", {"lens": "sigma"}),
        ("food last month", "food", {"date_from": date(2026, 8, 1), "date_to": date(2026, 8, 31)}),
    ],
)
def test_rule_parser(q, semantic, expect):
    p = parse(q)
    got = p.filters.model_dump(exclude_none=True)
    got = {k: v for k, v in got.items() if v != []}
    if "people" in got:
        got["people"] = sorted(got["people"], key=str.lower)
        expect = {**expect, "people": sorted(expect["people"], key=str.lower)}
    assert got == expect, q
    assert p.semantic.lower() == semantic.lower(), q


def test_prime_regex_does_not_match_zoom():
    from api.search.filters import SearchFilters, to_sql

    _sql, params = to_sql(SearchFilters(lens="50mm"))
    import re

    rx = re.compile(params["f_lens"].replace("(^|", "(?:^|"), re.I)
    assert rx.search("RF50mm F1.8 STM")
    assert rx.search("EF50mm")
    assert not rx.search("EF-S 18-150mm")
    assert not rx.search("150mm macro")
