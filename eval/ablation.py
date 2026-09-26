"""Ablation and model-comparison tables for the README.

    uv run python -m eval.ablation                    # signals, dev split
    uv run python -m eval.ablation --split test       # final numbers
    uv run python -m eval.ablation --models openclip-vitb32,openclip-vitl14,finetuned-v1

Every row is a full eval run (own report + MLflow run), so each number in the
table links back to the run that produced it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from api.config import get_settings
from eval.report import ablation_table
from eval.run_eval import EvalConfig, run_eval
from eval.tracking import REPORTS_DIR

EQUAL = {"clip": 1.0, "caption": 1.0, "keyword": 1.0, "ocr": 1.0}

SIGNAL_ABLATION: list[tuple[str, dict[str, Any]]] = [
    ("CLIP only", dict(signals=["clip"], parse="off")),
    ("CLIP + query parser", dict(signals=["clip"], parse="auto")),
    (
        "+ captions (embedding + keywords)",
        dict(signals=["clip", "caption", "keyword"], weights=EQUAL),
    ),
    ("+ OCR", dict(signals=["clip", "caption", "keyword", "ocr"], weights=EQUAL)),
    ("+ tuned fusion weights", dict(signals=None, weights=None)),
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default="dev")
    ap.add_argument("--models", help="comma-separated image models to compare instead of signals")
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--queries-file")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()

    configs: list[tuple[str, dict[str, Any]]]
    if a.models:
        configs = [(m, dict(model=m, signals=["clip"], parse="auto")) for m in a.models.split(",")]
        out_name = f"models_{a.split}.md"
    else:
        if not get_settings().fusion_weights_file.exists():
            print("note: no tuned weights yet (run `make tune-fusion`); last row uses defaults")
        configs = SIGNAL_ABLATION
        out_name = f"ablation_{a.split}.md"

    reports = []
    for label, kw in configs:
        cfg = EvalConfig(
            name=f"ablation-{label}", split=a.split, mlflow=not a.no_mlflow,
            queries_file=a.queries_file, **kw,
        )  # fmt: skip
        rep = run_eval(cfg)
        reports.append((label, rep))
        print(f"{label:40s} {a.metric}={rep['overall'][a.metric]:.3f}")
    md = ablation_table(reports, a.metric)
    other = "recall@20" if a.metric != "recall@20" else "ndcg@10"
    md += "\n" + ablation_table(reports, other)
    path: Path = REPORTS_DIR / out_name
    path.write_text(md)
    print("\n" + md)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
