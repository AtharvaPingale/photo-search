"""Weighted reciprocal rank fusion.

RRF only looks at ranks, so it doesn't care that CLIP cosine similarities sit
in [0.15, 0.35] while ts_rank_cd scores are unbounded. Each signal contributes
w / (k + rank); a photo missing from a signal's list contributes nothing.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence


def rrf(
    rankings: Mapping[str, Sequence[Hashable]],
    weights: Mapping[str, float] | None = None,
    k: int = 60,
) -> list[tuple[Hashable, float, dict[str, int]]]:
    """Fuse ranked lists -> [(id, score, {signal: 1-based rank})], best first.

    Ties break on the best single rank, then on id, so results are deterministic.
    """
    weights = weights or {}
    scores: dict[Hashable, float] = {}
    ranks: dict[Hashable, dict[str, int]] = {}
    for signal, ids in rankings.items():
        w = weights.get(signal, 1.0)
        if w <= 0:
            continue
        for r, pid in enumerate(ids, start=1):
            scores[pid] = scores.get(pid, 0.0) + w / (k + r)
            ranks.setdefault(pid, {})[signal] = r
    return sorted(
        ((pid, s, ranks[pid]) for pid, s in scores.items()),
        key=lambda t: (-t[1], min(t[2].values()), str(t[0])),
    )
