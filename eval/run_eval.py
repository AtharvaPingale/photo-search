"""Retrieval eval: Recall@5, Recall@20, MRR, nDCG@10 per category, plus p50/p95 latency.

    uv run python -m eval.run_eval --name baseline            # dev split
    uv run python -m eval.run_eval --name final --split test  # held-out, final numbers only

Writes eval/reports/<timestamp>_<name>_<split>.{json,md}, logs the run to
MLflow (params, per-category metrics, git commit, report artifacts), and
prints a comparison against the previous run on the same split.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.dataset import EvalQuery, load_queries
from eval.metrics import METRICS, bootstrap_ci, mean_metrics, percentile, query_metrics
from eval.report import latest_report, load_report, render_markdown
from eval.tracking import REPORTS_DIR, git_commit, mlflow_run


@dataclass
class EvalConfig:
    name: str = "run"
    split: str = "dev"
    model: str | None = None
    signals: list[str] | None = None
    weights: dict[str, float] | None = None
    parse: str = "auto"
    k: int = 100
    queries_file: str | None = None
    mlflow: bool = True
    compare: str | None = None
    write: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def hash_to_ids(hashes: set[str]) -> dict[str, list]:
    from api.db.session import get_conn

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, file_hash FROM photos WHERE file_hash = ANY(%s)", (list(hashes),)
        ).fetchall()
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r["file_hash"], []).append(r["id"])
    return out


def evaluate_queries(
    queries: list[EvalQuery], cfg: EvalConfig
) -> tuple[list[dict], dict[str, list[float]]]:
    from api.search.engine import SearchRequest, search

    lat: dict[str, list[float]] = {"total": [], "retrieval": [], "parse": []}
    rows = []

    def req(q: str) -> SearchRequest:
        return SearchRequest(
            q=q, k=cfg.k, parse=cfg.parse, model=cfg.model, signals=cfg.signals,
            weights=cfg.weights, group_videos=False,
        )  # fmt: skip

    if queries:  # warm-up: model load and first-query costs aren't query latency
        search(req(queries[0].query))
    for q in queries:
        resp = search(req(q.query))
        t = resp.timings_ms
        lat["total"].append(t.get("total", math.nan))
        lat["retrieval"].append(t.get("encode", 0) + t.get("retrieve", 0))
        lat["parse"].append(t.get("parse", 0))
        ranked = [h.file_hash for h in resp.hits]
        judg = q.judgments()
        first = next((i for i, h in enumerate(ranked, 1) if h in judg), None)
        rows.append(
            {
                "id": q.id,
                "query": q.query,
                "category": q.category,
                "n_relevant": len(judg),
                "metrics": query_metrics(ranked, judg),
                "first_hit_rank": first,
                "top10": ranked[:10],
                "parsed": {
                    "semantic": resp.parsed.semantic,
                    "filters": resp.parsed.filters.model_dump(mode="json", exclude_none=True),
                    "source": resp.parsed.source,
                },
                "fallback_used": resp.fallback_used,
                "signals": resp.signals,
                "latency_ms": t.get("total"),
            }
        )
    return rows, lat


def run_eval(cfg: EvalConfig) -> dict[str, Any]:
    from api.ml.clip import get_clip

    qfile = Path(cfg.queries_file) if cfg.queries_file else None
    queries = load_queries(qfile, split=cfg.split)
    if not queries:
        raise SystemExit(
            f"no labelled queries for split={cfg.split!r}. Label some in the UI (Eval tab) "
            "or add them to eval/queries/queries.jsonl"
        )
    all_hashes = {h for q in queries for h in q.relevant}
    present = hash_to_ids(all_hashes)
    missing = len(all_hashes - set(present))

    cfg.model = cfg.model or get_clip().spec.name
    t0 = time.time()
    rows, lat = evaluate_queries(queries, cfg)
    wall = time.time() - t0

    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["metrics"])
    report: dict[str, Any] = {
        "kind": "retrieval",
        "name": cfg.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": {k: v for k, v in asdict(cfg).items() if k not in ("mlflow", "write", "compare")},
        "n_queries": len(rows),
        "missing_labels": missing,
        "overall": mean_metrics([r["metrics"] for r in rows]),
        "by_category": {c: mean_metrics(ms) for c, ms in by_cat.items()},
        "ci95": {
            m: bootstrap_ci([r["metrics"][m] for r in rows]) for m in ("ndcg@10", "recall@20")
        },
        "latency_ms": {
            "total_p50": percentile(lat["total"], 50),
            "total_p95": percentile(lat["total"], 95),
            "retrieval_p50": percentile(lat["retrieval"], 50),
            "retrieval_p95": percentile(lat["retrieval"], 95),
            "parse_p50": percentile(lat["parse"], 50),
            "parse_p95": percentile(lat["parse"], 95),
        },
        "wall_s": round(wall, 2),
        "per_query": rows,
    }
    report["index_size"] = _index_size()

    prev_path = Path(cfg.compare) if cfg.compare else latest_report(cfg.split)
    prev = load_report(prev_path) if prev_path and prev_path.exists() else None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = REPORTS_DIR / f"{stamp}_{_slug(cfg.name)}_{cfg.split}"
    with mlflow_run("retrieval", f"{cfg.name}-{cfg.split}", enabled=cfg.mlflow) as run:
        report["mlflow_run_id"] = run.run_id
        report["mlflow_experiment_id"] = run.experiment_id
        run.log_params(
            {
                "model": cfg.model, "split": cfg.split, "signals": cfg.signals or "all",
                "weights": json.dumps(cfg.weights) if cfg.weights else "default",
                "parse": cfg.parse, "k": cfg.k, "git_commit": report["git_commit"],
                "n_queries": len(rows), **{f"extra.{k}": v for k, v in cfg.extra.items()},
            }
        )  # fmt: skip
        metrics = {f"{m}": report["overall"][m] for m in METRICS}
        for c, cm in report["by_category"].items():
            metrics.update({f"{m}.{c}": cm[m] for m in METRICS})
        metrics.update({f"latency.{k}": v for k, v in report["latency_ms"].items()})
        run.log_metrics(metrics)
        if cfg.write:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            base.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str) + "\n")
            base.with_suffix(".md").write_text(render_markdown(report, prev))
            run.log_artifact(base.with_suffix(".json"))
            run.log_artifact(base.with_suffix(".md"))
    report["report_path"] = str(base.with_suffix(".md")) if cfg.write else None
    report["_markdown"] = render_markdown(report, prev)
    return report


def _index_size() -> dict[str, Any]:
    from api.db.session import get_conn

    with get_conn() as conn:
        row = conn.execute(
            "SELECT (SELECT count(*) FROM photos WHERE media_type = 'image') AS photos, "
            "pg_size_pretty(pg_total_relation_size('image_embeddings')) AS embeddings_size"
        ).fetchone()
    return dict(row or {})


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in s)[:40]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", default="run")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--model")
    ap.add_argument("--signals", help="comma-separated subset of clip,caption,keyword,ocr")
    ap.add_argument("--weights", help='JSON, e.g. \'{"clip":1,"caption":0.5}\'')
    ap.add_argument("--parse", default="auto", choices=["auto", "llm", "rules", "off"])
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--queries-file")
    ap.add_argument("--compare", help="report JSON to compare against (default: previous run)")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args(argv)
    if a.split == "test":
        print("NOTE: test split. Use it for final numbers only; tune on dev.\n")
    rep = run_eval(
        EvalConfig(
            name=a.name,
            split=a.split,
            model=a.model,
            signals=a.signals.split(",") if a.signals else None,
            weights=json.loads(a.weights) if a.weights else None,
            parse=a.parse,
            k=a.k,
            queries_file=a.queries_file,
            mlflow=not a.no_mlflow,
            compare=a.compare,
        )
    )
    print(rep["_markdown"])
    if rep.get("report_path"):
        print(f"report: {rep['report_path']}")


if __name__ == "__main__":
    main()
