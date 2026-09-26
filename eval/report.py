"""Eval report rendering and run-to-run comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from eval.metrics import METRICS
from eval.tracking import REPORTS_DIR, tracking_uri


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def latest_report(split: str, exclude: Path | None = None, kind: str = "retrieval") -> Path | None:
    if not REPORTS_DIR.exists():
        return None
    cands = []
    for p in REPORTS_DIR.glob("*.json"):
        if exclude and p.resolve() == exclude.resolve():
            continue
        try:
            r = load_report(p)
        except (json.JSONDecodeError, OSError):
            continue
        if r.get("kind", "retrieval") == kind and r.get("config", {}).get("split") == split:
            cands.append((r.get("created_at", ""), p))
    return max(cands)[1] if cands else None


def _f(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{digits}f}"


def _delta(a: float | None, b: float | None) -> str:
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return ""
    d = b - a
    if abs(d) < 0.0005:
        return " (±0)"
    return f" ({'+' if d > 0 else ''}{d:.3f})"


def mlflow_link(report: dict[str, Any]) -> str:
    rid = report.get("mlflow_run_id")
    if not rid:
        return "no MLflow run"
    exp = report.get("mlflow_experiment_id", "0")
    return f"[MLflow run `{rid[:8]}`](http://localhost:5000/#/experiments/{exp}/runs/{rid}) ({tracking_uri()})"


def render_markdown(report: dict[str, Any], prev: dict[str, Any] | None = None) -> str:
    cfg = report["config"]
    lines = [
        f"# Eval: {report['name']} ({cfg['split']})",
        "",
        f"- commit `{report['git_commit']}` · {report['created_at']} · {mlflow_link(report)}",
        f"- model `{cfg.get('model')}` · signals `{cfg.get('signals') or 'all'}` · parser `{cfg.get('parse')}`"
        f" · weights `{cfg.get('weights') or 'default'}`",
        f"- {report['n_queries']} queries"
        + (
            f" · {report['missing_labels']} labelled photos not in the index"
            if report.get("missing_labels")
            else ""
        ),
    ]
    if prev:
        lines.append(
            f"- compared with **{prev['name']}** (`{prev['git_commit']}`, {prev['created_at']})"
        )
    lines += [
        "",
        "| category | n | " + " | ".join(METRICS) + " |",
        "|---|---:|" + "---:|" * len(METRICS),
    ]
    rows = [("**all**", report["overall"], (prev or {}).get("overall"))]
    for cat, m in sorted(report["by_category"].items()):
        rows.append((cat, m, (prev or {}).get("by_category", {}).get(cat)))
    for name, m, pm in rows:
        cells = [f"{_f(m[k])}{_delta(pm.get(k) if pm else None, m[k])}" for k in METRICS]
        lines.append(f"| {name} | {m['n']} | " + " | ".join(cells) + " |")
    ci = report.get("ci95", {})
    if ci:
        lines += [
            "",
            "95% bootstrap CI (all queries): "
            + ", ".join(f"{k} [{_f(a)}, {_f(b)}]" for k, (a, b) in ci.items()),
        ]
    lat = report["latency_ms"]
    pl = (prev or {}).get("latency_ms", {})
    lines += [
        "",
        "| latency (ms) | p50 | p95 |",
        "|---|---:|---:|",
        f"| end to end | {_f(lat['total_p50'], 1)}{_delta(pl.get('total_p50'), lat['total_p50'])} | "
        f"{_f(lat['total_p95'], 1)}{_delta(pl.get('total_p95'), lat['total_p95'])} |",
        f"| retrieval only (encode + SQL) | {_f(lat['retrieval_p50'], 1)} | {_f(lat['retrieval_p95'], 1)} |",
        f"| query parser | {_f(lat['parse_p50'], 1)} | {_f(lat['parse_p95'], 1)} |",
    ]
    worst = sorted(report["per_query"], key=lambda q: (q["metrics"]["ndcg@10"], q["id"]))[:10]
    lines += [
        "",
        "## Worst queries (nDCG@10)",
        "",
        "| id | category | query | nDCG@10 | R@20 | first hit |",
        "|---|---|---|---:|---:|---:|",
    ]
    for q in worst:
        fh = q.get("first_hit_rank")
        lines.append(
            f"| {q['id']} | {q['category']} | {q['query']} | {_f(q['metrics']['ndcg@10'])} | "
            f"{_f(q['metrics']['recall@20'])} | {fh if fh else '–'} |"
        )
    return "\n".join(lines) + "\n"


def ablation_table(reports: list[tuple[str, dict[str, Any]]], metric: str = "ndcg@10") -> str:
    """Rows = configurations, columns = categories. The README's ablation table."""
    cats = sorted({c for _, r in reports for c in r["by_category"]})
    head = "| configuration | all | " + " | ".join(cats) + " | p50 ms | run |"
    sep = "|---|---:|" + "---:|" * len(cats) + "---:|---|"
    lines = [f"**{metric}** ({reports[0][1]['config']['split']} split)", "", head, sep]
    for label, r in reports:
        cells = [_f(r["by_category"].get(c, {}).get(metric)) for c in cats]
        rid = (r.get("mlflow_run_id") or "")[:8] or "–"
        lines.append(
            f"| {label} | {_f(r['overall'][metric])} | "
            + " | ".join(cells)
            + f" | {_f(r['latency_ms']['total_p50'], 0)} | `{rid}` |"
        )
    return "\n".join(lines) + "\n"
