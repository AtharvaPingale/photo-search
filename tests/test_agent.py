from __future__ import annotations

import json
from pathlib import Path

from api.agent import graph, tools
from api.db.session import get_conn
from api.llm import ChatTurn, ToolCall


def test_aggregate_most_photographed_city(indexed):
    res, ids = tools.run_tool(
        "aggregate",
        {"group_by": "city", "filters": {"date_from": "2025-01-01", "date_to": "2025-12-31"}},
    )
    assert res["groups"][0] == {
        "value": "Chicago",
        "count": 2,
        "example_photo_id": res["groups"][0]["example_photo_id"],
    }
    assert res["groups"][1]["value"] == "Tokyo"
    assert ids  # example ids are recorded as evidence


def test_metadata_templates(indexed):
    res, ids = tools.run_tool(
        "metadata_query", {"template": "first_and_last", "params": {"place": "Chicago"}}
    )
    assert res["total_matching"] == 2
    assert res["last"]["taken_at"].startswith("2025-06-02")
    assert len(ids) == 2
    res, _ = tools.run_tool("metadata_query", {"template": "library_overview", "params": {}})
    assert res["photos"] == "6"


def test_tools_reject_bad_input(indexed):
    res, _ = tools.run_tool("aggregate", {"group_by": "p.id; DROP TABLE photos"})
    assert res["error"] == "invalid arguments"
    res, _ = tools.run_tool(
        "metadata_query", {"template": "raw_sql", "params": {"sql": "DELETE FROM photos"}}
    )
    assert res["error"] == "invalid arguments"
    res, _ = tools.run_tool("nope", {})
    assert "unknown tool" in res["error"]
    # hostile strings are just bound parameters
    res, _ = tools.run_tool("aggregate", {"group_by": "city", "filters": {"place": "x' OR '1'='1"}})
    assert res["total_photos_matching_filters"] == 0
    with get_conn() as conn:
        assert conn.execute("SELECT count(*) n FROM photos").fetchone()["n"] == 6


def test_semantic_search_sort_newest(indexed):
    res, _ = tools.run_tool("semantic_search", {"query": "red", "sort": "newest", "limit": 2})
    dates = [p["taken_at"] for p in res["photos"]]
    assert dates == sorted(dates, reverse=True)


def test_graph_grounds_citations(indexed, monkeypatch):
    """Scripted LLM: one aggregate call, then a final answer citing one real and one invented id."""
    turns = iter(
        [
            ChatTurn("", [ToolCall("c1", "aggregate", {"group_by": "city"})]),
            ChatTurn("Chicago is your most photographed city."),
        ]
    )
    monkeypatch.setattr(graph, "chat_with_tools", lambda *a, **k: next(turns))

    def fake_final(system, user, schema, **kw):
        real = json.loads(user.split("[aggregate] ", 1)[1].split("\n")[0])["groups"][0][
            "example_photo_id"
        ]
        return {
            "answer": "Chicago.",
            "evidence": [
                {"photo_id": real, "note": "Chicago"},
                {"photo_id": "00000000-0000-0000-0000-000000000000", "note": "made up"},
            ],
        }

    monkeypatch.setattr(graph, "complete_json", fake_final)
    graph._graph = None
    ans = graph.run_agent("What is my most photographed city?")
    assert ans.answer == "Chicago."
    assert ans.tool_calls == 1
    assert len(ans.evidence) == 1
    assert ans.invalid_citations == ["00000000-0000-0000-0000-000000000000"]
    assert not ans.grounded


def test_graph_stops_at_tool_budget(indexed, monkeypatch):
    monkeypatch.setattr(
        graph, "chat_with_tools",
        lambda *a, **k: ChatTurn("", [ToolCall("c", "metadata_query", {"template": "library_overview"})]),
    )  # fmt: skip
    monkeypatch.setattr(
        graph, "complete_json", lambda *a, **k: {"answer": "stopped", "evidence": []}
    )
    graph._graph = None
    ans = graph.run_agent("loop forever")
    assert ans.tool_calls == graph.MAX_TOOL_CALLS
    assert ans.answer == "stopped"
    _ = Path


def test_inline_citations_are_harvested_and_checked(indexed, monkeypatch):
    turns = iter(
        [ChatTurn("", [ToolCall("c1", "aggregate", {"group_by": "city"})]), ChatTurn("done")]
    )
    monkeypatch.setattr(graph, "chat_with_tools", lambda *a, **k: next(turns))

    def fake_final(system, user, schema, **kw):
        real = json.loads(user.split("[aggregate] ", 1)[1].split("\n")[0])["groups"][0][
            "example_photo_id"
        ]
        return {"answer": f"Chicago (Photo ID: {real}).", "evidence": []}

    monkeypatch.setattr(graph, "complete_json", fake_final)
    graph._graph = None
    ans = graph.run_agent("most photographed city?")
    assert len(ans.evidence) == 1 and ans.grounded
