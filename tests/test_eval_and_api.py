from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.db.session import get_conn
from eval import dataset
from eval.metrics import mrr, ndcg_at_k, recall_at_k


def test_metrics():
    rel = {"a": 1, "b": 2}
    assert recall_at_k(["x", "a", "y", "b"], rel, 2) == 0.5
    assert mrr(["x", "a"], rel) == 0.5
    assert mrr(["x"], rel) == 0.0
    assert ndcg_at_k(["b", "a"], rel, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "b"], rel, 10) < 1.0
    assert math.isnan(recall_at_k(["a"], {}, 5))


def test_split_is_stable_and_roughly_70_30():
    splits = [dataset.assign_split(f"q{i:04d}") for i in range(1000)]
    assert splits == [dataset.assign_split(f"q{i:04d}") for i in range(1000)]
    assert 0.25 < splits.count("test") / 1000 < 0.35


@pytest.fixture
def qfile(tmp_path, monkeypatch) -> Path:
    p = tmp_path / "queries.jsonl"
    monkeypatch.setattr(dataset, "QUERIES_FILE", p)
    return p


def _hashes() -> dict[str, str]:
    with get_conn() as conn:
        return {
            Path(r["path"]).name: r["file_hash"]
            for r in conn.execute("SELECT path, file_hash FROM photos")
        }


def test_run_eval_end_to_end(indexed, qfile, tmp_path, monkeypatch):
    from eval import report, run_eval, tracking

    monkeypatch.setattr(tracking, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(run_eval, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    h = _hashes()
    qs = [
        dataset.EvalQuery(id="q0001", query="red", category="objects", split="dev", relevant=[h["red_car.jpg"], h["red_sign.jpg"]]),
        dataset.EvalQuery(id="q0002", query="photos from Tokyo", category="metadata", split="dev", relevant=[h["blue_night.jpg"]]),
        dataset.EvalQuery(id="q0003", query="green", category="objects", split="dev", relevant=[h["green_park.jpg"]], grades={h["green_park.jpg"]: 2}),
        dataset.EvalQuery(id="q0004", query="unlabelled", category="objects", split="dev"),
    ]  # fmt: skip
    dataset.save_queries(qs, qfile)
    cfg = run_eval.EvalConfig(name="t1", split="dev", parse="rules", mlflow=False)
    rep = run_eval.run_eval(cfg)
    assert rep["n_queries"] == 3  # unlabelled query skipped
    assert rep["overall"]["recall@5"] == 1.0
    assert rep["by_category"]["metadata"]["mrr"] == 1.0
    assert rep["latency_ms"]["total_p50"] > 0
    md = Path(rep["report_path"]).read_text()
    assert "| **all** |" in md and "Worst queries" in md
    # second run compares against the first
    rep2 = run_eval.run_eval(run_eval.EvalConfig(name="t2", split="dev", parse="off", mlflow=False))
    assert "compared with **t1**" in rep2["_markdown"]
    assert rep2["by_category"]["metadata"]["mrr"] < 1.0  # no parser -> Tokyo filter lost


def test_tune_fusion_on_dev(indexed, qfile, tmp_path, monkeypatch):
    from api.config import get_settings
    from eval import tune_fusion

    h = _hashes()
    dataset.save_queries(
        [dataset.EvalQuery(id=f"q{i:04d}", query=q, split="dev", relevant=[h[n]]) for i, (q, n) in enumerate(
            [("red", "red_car.jpg"), ("blue", "blue_night.jpg"), ("green", "green_park.jpg"),
             ("yellow", "yellow_flower.jpg"), ("white", "white_wall.jpg")], 1)],
        qfile,
    )  # fmt: skip
    monkeypatch.setattr(get_settings(), "fusion_weights_file", tmp_path / "w.json")
    out = tune_fusion.tune(parse="off", folds=5, mlflow=False)
    assert out["weights"]["clip"] == 1.0
    assert json.loads((tmp_path / "w.json").read_text())["n_queries"] == 5


def test_api(indexed, qfile):
    from api.main import create_app

    client = TestClient(create_app())
    assert client.get("/api/health").json() == {"ok": True}

    r = client.get("/api/search", params={"q": "red", "k": 2, "parse": "off"}).json()
    assert len(r["hits"]) == 2 and r["timings_ms"]["total"] > 0
    pid = r["hits"][0]["photo_id"]
    assert client.get(f"/api/photos/{pid}/thumb").headers["content-type"] == "image/jpeg"
    d = client.get(f"/api/photos/{pid}").json()
    assert d["camera"] == "Canon EOS R6" and "openclip" not in d["embedding_models"]
    assert client.get(f"/api/photos/{pid}/original").status_code == 200
    assert client.get("/api/photos/00000000-0000-0000-0000-000000000000").status_code == 404

    p = client.get(
        "/api/parse", params={"q": "sunset in Tokyo last summer", "mode": "rules"}
    ).json()
    assert p["filters"]["place"] == "Tokyo" and p["semantic"] == "sunset"

    r = client.post(
        "/api/search", json={"q": "x", "parse": "off", "filters": {"place": "Paris"}}
    ).json()
    assert [Path(h["path"]).name for h in r["hits"]] == ["green_park.jpg"]
    r = client.get(f"/api/search/similar/{pid}", params={"k": 1}).json()
    assert r["hits"][0]["photo_id"] != pid

    with open(indexed / "2025/tokyo/blue_night.jpg", "rb") as f:
        r = client.post(
            "/api/search/image", files={"file": ("x.jpg", f, "image/jpeg")}, params={"k": 1}
        ).json()
    assert Path(r["hits"][0]["path"]).name == "blue_night.jpg"

    # labelling tool
    cands = client.post("/api/eval/candidates", json={"query": "red", "per_system": 5}).json()
    assert cands["n"] >= 2 and "found_by" in cands["candidates"][0]
    saved = client.post("/api/eval/queries", json={"query": "red", "category": "objects",
                                                    "relevant": [cands["candidates"][0]["file_hash"]]}).json()  # fmt: skip
    assert saved["id"] == "q0001" and saved["split"] in ("dev", "test")
    listing = client.get("/api/eval/queries").json()
    assert listing["counts"]["labeled"] == 1
    assert client.delete("/api/eval/queries/q0001").json() == {"deleted": True}

    assert client.get("/api/stats").json()["photos"] == 6
    assert "Chicago" in client.get("/api/vocab").json()["places"]
    assert client.get("/api/groups/burst").json() == []


def test_tuner_clip_only_baseline_really_is_clip_only():
    from eval.dataset import EvalQuery
    from eval.tune_fusion import score

    q = EvalQuery(id="q1", query="x", relevant=["b"])
    data = [(q, {"clip": ["a", "b"], "caption": ["b", "a"]})]
    clip_only = score(data, {"clip": 1.0, "caption": 0.0}, 60, "mrr")
    assert clip_only == 0.5  # b is second in CLIP's ranking
    assert score(data, {"clip": 1.0, "caption": 2.0}, 60, "mrr") == 1.0  # caption outvotes CLIP
