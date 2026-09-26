"""Near-duplicate detection: precision and recall on labelled pairs.

    uv run python -m eval.dup_eval sample --n 150    # candidate pairs -> eval/queries/dup_pairs.csv
    uv run python -m eval.dup_eval score

The sample mixes pairs the detector flagged with near-misses just below the
CLIP threshold, so recall is measurable too (a sample of flagged pairs alone can
only measure precision). Mark duplicate = y/n.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import UTC, datetime
from typing import Any

import numpy as np

from api.config import ROOT
from api.db.session import get_conn
from eval.tracking import REPORTS_DIR, git_commit, mlflow_run

PAIRS = ROOT / "eval" / "queries" / "dup_pairs.csv"


def predicted_groups() -> dict[str, int]:
    with get_conn() as conn:
        return {
            r["file_hash"]: r["group_id"]
            for r in conn.execute(
                "SELECT p.file_hash, g.group_id FROM photo_groups g JOIN photos p ON p.id = g.photo_id "
                "WHERE g.kind = 'duplicate'"
            )
        }


def sample(n: int = 150, seed: int = 0) -> int:
    from api.ml.clip import get_clip
    from workers.embed import load_embeddings
    from workers.organize import topk_neighbours

    ids, X = load_embeddings(get_clip().spec.name, "AND NOT p.is_video_frame")
    with get_conn() as conn:
        meta = {
            r["id"]: r
            for r in conn.execute(
                "SELECT id, file_hash, path FROM photos WHERE id = ANY(%s)", (ids,)
            )
        }
    groups = predicted_groups()
    sims, idx = topk_neighbours(X, k=5)
    flagged, near = [], []
    for i in range(len(ids)):
        for s, j in zip(sims[i], idx[i], strict=True):
            if j <= i:
                continue
            a, b = meta[ids[i]], meta[ids[int(j)]]
            pair = (a["file_hash"], b["file_hash"], a["path"], b["path"], float(s))
            ga, gb = groups.get(a["file_hash"]), groups.get(b["file_hash"])
            if ga is not None and ga == gb:
                flagged.append(pair)
            elif 0.88 <= s < 0.97:
                near.append(pair)
    rng = random.Random(seed)
    pick = rng.sample(flagged, min(len(flagged), n // 2))
    pick += rng.sample(near, min(len(near), n - len(pick)))
    rng.shuffle(pick)
    with PAIRS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hash_a", "hash_b", "path_a", "path_b", "clip_sim", "duplicate"])
        for p in pick:
            w.writerow([*p[:4], f"{p[4]:.4f}", ""])
    return len(pick)


def score(mlflow: bool = True) -> dict[str, Any]:
    groups = predicted_groups()
    tp = fp = fn = tn = 0
    with PAIRS.open() as f:
        for row in csv.DictReader(f):
            lab = (row.get("duplicate") or "").strip().lower()
            if lab not in ("y", "n"):
                continue
            ga, gb = groups.get(row["hash_a"]), groups.get(row["hash_b"])
            pred = ga is not None and ga == gb
            gold = lab == "y"
            tp += pred and gold
            fp += pred and not gold
            fn += gold and not pred
            tn += not pred and not gold
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    out = {
        "kind": "duplicates", "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r,
        "f1": float(2 * p * r / (p + r)) if (tp and p + r) else float("nan"),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "git_commit": git_commit(),
    }  # fmt: skip
    with mlflow_run("duplicates", "pairs", enabled=mlflow) as run:
        run.log_metrics({k: v for k, v in out.items() if isinstance(v, float) and not np.isnan(v)})
        out["mlflow_run_id"] = run.run_id
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_duplicates.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=150)
    sc = sub.add_parser("score")
    sc.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    if a.cmd == "sample":
        print(f"wrote {sample(a.n)} pairs to {PAIRS}; fill in duplicate = y/n")
    else:
        print(json.dumps(score(not a.no_mlflow), indent=2))


if __name__ == "__main__":
    main()
