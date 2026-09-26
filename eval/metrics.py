"""Retrieval metrics. `ranked` is the system's ordered list of ids; `relevant`
maps id -> graded relevance (1 = relevant, 2 = highly relevant)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def recall_at_k(ranked: Sequence[Any], relevant: Mapping[Any, int], k: int) -> float:
    if not relevant:
        return float("nan")
    hits = sum(1 for x in ranked[:k] if x in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: Sequence[Any], relevant: Mapping[Any, int], k: int) -> float:
    if k <= 0:
        return float("nan")
    return sum(1 for x in ranked[:k] if x in relevant) / k


def mrr(ranked: Sequence[Any], relevant: Mapping[Any, int], cutoff: int = 100) -> float:
    for i, x in enumerate(ranked[:cutoff], start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[Any], relevant: Mapping[Any, int], k: int) -> float:
    """Graded nDCG with gain 2^rel - 1 (so a grade-2 photo counts 3x a grade-1 one)."""
    if not relevant:
        return float("nan")
    dcg = sum(
        (2 ** relevant.get(x, 0) - 1) / math.log2(i + 1) for i, x in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def query_metrics(ranked: Sequence[Any], relevant: Mapping[Any, int]) -> dict[str, float]:
    return {
        "recall@5": recall_at_k(ranked, relevant, 5),
        "recall@20": recall_at_k(ranked, relevant, 20),
        "mrr": mrr(ranked, relevant),
        "ndcg@10": ndcg_at_k(ranked, relevant, 10),
    }


METRICS = ("recall@5", "recall@20", "mrr", "ndcg@10")


def mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    out = {}
    for m in METRICS:
        vals = [r[m] for r in rows if not math.isnan(r[m])]
        out[m] = float(np.mean(vals)) if vals else float("nan")
    out["n"] = len(rows)
    return out


def percentile(values: Sequence[float], p: float) -> float:
    return float(np.percentile(values, p)) if len(values) else float("nan")


def bootstrap_ci(
    values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap CI of the mean. With ~40 test queries, deltas under ~0.03 are noise;
    reports show the interval so nobody over-reads a small change."""
    v = np.asarray([x for x in values if not math.isnan(x)], float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_p(
    a: Sequence[float], b: Sequence[float], n: int = 5000, seed: int = 0
) -> float:
    """One-sided p-value that system b is not better than a on the same queries."""
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float((means <= 0).mean())
