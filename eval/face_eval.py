"""Face clustering quality on a hand-labelled sample.

    uv run python -m eval.face_eval sample --n 200   # writes eval/queries/face_labels.csv to fill in
    uv run python -m eval.face_eval score

Label a face with the person's name, or "?" if you can't tell (excluded).
Faces are sampled across clusters and from the unclustered noise so both
kinds of error show up.

Metrics
  purity      share of clustered, labelled faces that match their cluster's majority person
  completeness  share of each person's clustered faces that sit in that person's largest cluster
  coverage    share of labelled faces that got a cluster at all (HDBSCAN leaves noise unassigned)
  pairwise P/R/F1  over all pairs of labelled faces: "same cluster" vs "same person"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

from api.config import ROOT
from api.db.session import get_conn
from api.paths import resolve
from eval.tracking import REPORTS_DIR, git_commit, mlflow_run

LABELS = ROOT / "eval" / "queries" / "face_labels.csv"


def sample(n: int = 200, seed: int = 0) -> int:
    with get_conn() as conn:
        rows = conn.execute("SELECT id, cluster_id, crop_path FROM faces ORDER BY id").fetchall()
    rng = random.Random(seed)
    by_c: dict[Any, list] = {}
    for r in rows:
        by_c.setdefault(r["cluster_id"], []).append(r)
    picked = []
    # stratified: a few faces from every cluster (and from noise), then fill randomly
    for faces in by_c.values():
        picked += rng.sample(faces, min(len(faces), max(2, n // max(1, len(by_c)))))
    rest = [r for r in rows if r not in picked]
    picked += rng.sample(rest, max(0, min(len(rest), n - len(picked))))
    rng.shuffle(picked)
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with LABELS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["face_id", "crop_path", "person"])
        for r in picked[:n]:
            w.writerow([r["id"], resolve(r["crop_path"]), ""])
    return min(n, len(picked))


def clustering_metrics(assign: dict[str, int | None], truth: dict[str, str]) -> dict[str, float]:
    faces = [f for f in truth if f in assign]
    clustered = [f for f in faces if assign[f] is not None]
    coverage = len(clustered) / len(faces) if faces else float("nan")
    by_cluster: dict[int, list[str]] = {}
    for f in clustered:
        by_cluster.setdefault(assign[f], []).append(truth[f])  # type: ignore[arg-type]
    purity = (
        sum(Counter(v).most_common(1)[0][1] for v in by_cluster.values()) / len(clustered)
        if clustered
        else float("nan")
    )
    by_person: dict[str, list[int]] = {}
    for f in clustered:
        by_person.setdefault(truth[f], []).append(assign[f])  # type: ignore[arg-type]
    completeness = (
        sum(Counter(v).most_common(1)[0][1] for v in by_person.values()) / len(clustered)
        if clustered
        else float("nan")
    )
    tp = fp = fn = 0
    for a, b in combinations(faces, 2):
        same_c = assign[a] is not None and assign[a] == assign[b]
        same_p = truth[a] == truth[b]
        tp += same_c and same_p
        fp += same_c and not same_p
        fn += same_p and not same_c
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return {
        "n_labelled": len(faces), "coverage": coverage, "purity": purity, "completeness": completeness,
        "pairwise_precision": p, "pairwise_recall": r, "pairwise_f1": f1,
        "n_clusters_touched": len(by_cluster), "n_people": len(by_person),
    }  # fmt: skip


def score(mlflow: bool = True) -> dict[str, Any]:
    truth = {}
    with LABELS.open() as f:
        for row in csv.DictReader(f):
            name = (row.get("person") or "").strip()
            if name and name != "?":
                truth[row["face_id"]] = name.lower()
    with get_conn() as conn:
        assign = {
            str(r["id"]): r["cluster_id"] for r in conn.execute("SELECT id, cluster_id FROM faces")
        }
    m: dict[str, Any] = dict(clustering_metrics(assign, truth))
    m.update(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        kind="faces",
    )
    with mlflow_run("faces", "clustering", enabled=mlflow) as run:
        run.log_metrics({k: v for k, v in m.items() if isinstance(v, float)})
        m["mlflow_run_id"] = run.run_id
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_faces.json").write_text(
        json.dumps(m, indent=2) + "\n"
    )
    return m


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=200)
    sc = sub.add_parser("score")
    sc.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    if a.cmd == "sample":
        print(f"wrote {sample(a.n)} faces to {LABELS}; fill in the 'person' column")
    else:
        print(json.dumps(score(not a.no_mlflow), indent=2))


if __name__ == "__main__":
    main()
