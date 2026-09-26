from __future__ import annotations

from datetime import date
from pathlib import Path

from api.db.session import get_conn
from api.search.engine import SearchRequest, search
from api.search.filters import SearchFilters
from api.search.fusion import rrf


def names(resp) -> list[str]:
    return [Path(h.path).name for h in resp.hits]


def test_rrf_fuses_and_is_deterministic():
    out = rrf({"a": ["x", "y", "z"], "b": ["y", "x"]}, {"a": 1.0, "b": 1.0}, k=60)
    assert [pid for pid, _, _ in out][:2] in (["x", "y"], ["y", "x"])
    assert out[0][1] == out[1][1]  # symmetric tie
    assert [pid for pid, _, _ in out][2] == "z"
    weighted = rrf({"a": ["x", "y"], "b": ["y", "x"]}, {"a": 2.0, "b": 1.0})
    assert weighted[0][0] == "x"
    assert rrf({"a": ["x"]}, {"a": 0.0}) == []


def test_semantic_search_ranks_by_colour(indexed):
    resp = search(SearchRequest(q="red", k=3, parse="off"))
    assert set(names(resp)[:2]) == {"red_car.jpg", "red_sign.jpg"}
    assert resp.signals == ["clip"]  # no captions or OCR in this library yet
    assert resp.timings_ms["total"] > 0


def test_filters_apply_in_sql(indexed):
    resp = search(SearchRequest(q="red", parse="off", filters=SearchFilters(place="Tokyo")))
    assert names(resp) == ["blue_night.jpg"]
    resp = search(
        SearchRequest(
            q="photo",
            parse="off",
            filters=SearchFilters(date_from=date(2024, 1, 1), date_to=date(2024, 12, 31)),
        )
    )
    assert set(names(resp)) == {"green_park.jpg", "yellow_flower.jpg"}
    resp = search(SearchRequest(q="x", parse="off", filters=SearchFilters(lens="50mm")))
    assert set(names(resp)) == {"red_car.jpg", "red_sign.jpg", "white_wall.jpg"}  # not the 18-150mm
    resp = search(SearchRequest(q="x", parse="off", filters=SearchFilters(iso_min=3200)))
    assert names(resp) == ["blue_night.jpg"]
    resp = search(SearchRequest(q="x", parse="off", filters=SearchFilters(place="united states")))
    assert set(names(resp)) == {"red_car.jpg", "red_sign.jpg"}


def test_rule_parser_end_to_end(indexed):
    resp = search(SearchRequest(q="red photos from Chicago", parse="rules"))
    assert resp.parsed.filters.place == "Chicago"
    assert resp.parsed.semantic == "red"
    assert set(names(resp)) == {"red_car.jpg", "red_sign.jpg"}

    resp = search(SearchRequest(q="portraits with the 85mm", parse="rules"))
    assert resp.parsed.filters.lens == "85mm"
    assert names(resp) == ["green_park.jpg"]


def test_metadata_only_query_orders_by_recency(indexed):
    resp = search(SearchRequest(q="photos from Chicago", parse="rules"))
    assert resp.parsed.semantic == ""
    assert names(resp) == ["red_sign.jpg", "red_car.jpg"]
    assert resp.signals == ["recency"]


def test_fallback_when_filters_match_nothing(indexed):
    resp = search(
        SearchRequest(q="red", parse="off", filters=SearchFilters(place="Paris", iso_min=100000))
    )
    assert resp.fallback_used
    assert names(resp)[0] in {"red_car.jpg", "red_sign.jpg"}


def test_search_by_example_and_feedback(indexed):
    with get_conn() as conn:
        ids = {
            Path(r["path"]).name: r["id"]
            for r in conn.execute("SELECT id, path FROM photos").fetchall()
        }
    resp = search(SearchRequest(like_photo_id=ids["red_car.jpg"], k=2))
    assert names(resp) == ["red_sign.jpg", names(resp)[1]]
    assert "red_car.jpg" not in names(resp)

    # "less like this": the negative photo disappears and similar ones sink
    resp = search(SearchRequest(q="red", parse="off", negative_ids=[ids["red_sign.jpg"]], k=6))
    assert "red_sign.jpg" not in names(resp)
    resp = search(
        SearchRequest(q="white", parse="off", positive_ids=[ids["yellow_flower.jpg"]], k=2)
    )
    assert "yellow_flower.jpg" in names(resp)


def test_pagination(indexed):
    full = names(search(SearchRequest(q="red", parse="off", k=6)))
    page2 = names(search(SearchRequest(q="red", parse="off", k=2, offset=2)))
    assert page2 == full[2:4]
