"""Tune RRF fusion weights on the dev split (never on test).

Per-signal rankings are computed once per query; every weight combination is
then just a re-fusion in Python, so a few hundred combinations take seconds.
The score reported for the chosen weights is a 5-fold cross-validated nDCG@10
on dev (weights picked on 4/5 of dev, scored on the held-out 1/5), which is an
honest estimate; the in-sample best is always optimistic.

    uv run python -m eval.tune_fusion
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from api.config import get_settings
from api.search.fusion import rrf
from eval.dataset import EvalQuery, load_queries
from eval.metrics import mean_metrics, query_metrics
from eval.tracking import git_commit, mlflow_run

GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def collect_rankings(
    queries: list[EvalQuery], *, parse: str = "auto", model: str | None = None
) -> list[tuple[EvalQuery, dict[str, list[str]]]]:
    from api.db.session import get_conn
    from api.search.engine import SearchRequest, compute_rankings
    from api.search.filters import SearchFilters
    from api.search.query_parser import parse_query

    out = []
    with get_conn() as conn:
        for q in queries:
            parsed = parse_query(q.query, parse)  # type: ignore[arg-type]
            req = SearchRequest(q=q.query, k=100, model=model)
            ranks = compute_rankings(conn, req, parsed)
            if not any(ranks.values()) and not parsed.filters.is_empty():
                parsed = parsed.model_copy(update={"semantic": q.query, "filters": SearchFilters()})
                ranks = compute_rankings(conn, req, parsed)
            ids = {pid for lst in ranks.values() for pid in lst}
            hashes = {
                r["id"]: r["file_hash"]
                for r in conn.execute(
                    "SELECT id, file_hash FROM photos WHERE id = ANY(%s)", (list(ids),)
                )
            }
            out.append(
                (q, {s: [hashes[p] for p in lst if p in hashes] for s, lst in ranks.items()})
            )
            conn.rollback()
    return out


def score(
    data: list[tuple[EvalQuery, dict[str, list[str]]]],
    weights: dict[str, float],
    k: int,
    metric: str,
) -> float:
    vals = []
    for q, ranks in data:
        fused = [pid for pid, _, _ in rrf(ranks, weights | {"recency": 1.0}, k=k)][:100]
        vals.append(query_metrics(fused, q.judgments())[metric])
    return float(np.nanmean(vals)) if vals else float("nan")


def grid(signals: list[str]) -> list[dict[str, float]]:
    # RRF is scale-invariant in the weights, so pin clip at 1.0
    others = [s for s in signals if s != "clip"]
    combos = []
    for ws in itertools.product(GRID, repeat=len(others)):
        combos.append({"clip": 1.0, **dict(zip(others, ws, strict=True))})
    return combos


def best(data, combos, k: int, metric: str) -> tuple[dict[str, float], float]:
    scored = [(score(data, w, k, metric), w) for w in combos]
    # tie-break toward smaller total weight on non-CLIP signals (simpler model)
    s, w = max(scored, key=lambda t: (round(t[0], 6), -sum(t[1].values())))
    return w, s


def tune(
    parse: str = "auto",
    queries_file: str | None = None,
    metric: str = "ndcg@10",
    folds: int = 5,
    rrf_ks=(60,),
    mlflow: bool = True,
) -> dict[str, Any]:
    queries = load_queries(Path(queries_file) if queries_file else None, split="dev")
    if len(queries) < folds:
        raise SystemExit("need labelled dev queries to tune on")
    data = collect_rankings(queries, parse=parse)
    signals = sorted({s for _, r in data for s in r if s != "recency"} | {"clip"})
    combos = grid(signals)

    results = {}
    for k in rrf_ks:
        w, s = best(data, combos, k, metric)
        # k-fold CV estimate of the tuning procedure itself
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(data))
        cv = []
        for f in range(folds):
            test_idx = set(idx[f::folds].tolist())
            train = [d for i, d in enumerate(data) if i not in test_idx]
            held = [d for i, d in enumerate(data) if i in test_idx]
            fw, _ = best(train, combos, k, metric)
            cv.append(score(held, fw, k, metric) * len(held))
        results[k] = (w, s, sum(cv) / len(data))
    k_best = max(results, key=lambda k: results[k][2])
    w, in_sample, cv_score = results[k_best]
    equal = {s: 1.0 for s in signals}
    # RRF treats a signal missing from the weights as weight 1.0, so zero the others explicitly
    clip_only = {sig: 0.0 for sig in signals} | {"clip": 1.0}
    out = {
        "weights": w,
        "rrf_k": k_best,
        "metric": metric,
        "dev_in_sample": in_sample,
        "dev_cv": cv_score,
        "dev_equal_weights": score(data, equal, k_best, metric),
        "dev_clip_only": score(data, clip_only, k_best, metric),
        "n_queries": len(data),
        "signals": signals,
        "tuned_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "per_category": _per_category(data, w, k_best),
    }
    with mlflow_run("fusion-tuning", f"tune-{metric}", enabled=mlflow) as run:
        run.log_params(
            {"parse": parse, "metric": metric, "rrf_k": k_best, "weights": json.dumps(w)}
        )
        run.log_metrics({k: v for k, v in out.items() if isinstance(v, float)})
        out["mlflow_run_id"] = run.run_id
    path = get_settings().fusion_weights_file
    path.write_text(json.dumps(out, indent=2) + "\n")
    return out


def _per_category(data, w, k) -> dict[str, dict[str, float]]:
    cats: dict[str, list] = {}
    for q, ranks in data:
        fused = [pid for pid, _, _ in rrf(ranks, w | {"recency": 1.0}, k=k)][:100]
        cats.setdefault(q.category, []).append(query_metrics(fused, q.judgments()))
    return {c: mean_metrics(v) for c, v in cats.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parse", default="auto")
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--rrf-k", default="60", help="comma-separated candidates, e.g. 20,60,100")
    ap.add_argument("--queries-file")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    out = tune(
        parse=a.parse,
        queries_file=a.queries_file,
        metric=a.metric,
        rrf_ks=[int(x) for x in a.rrf_k.split(",")],
        mlflow=not a.no_mlflow,
    )
    print(json.dumps({k: v for k, v in out.items() if k != "per_category"}, indent=2))
    print(f"wrote {get_settings().fusion_weights_file}")


if __name__ == "__main__":
    main()
