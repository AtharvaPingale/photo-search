"""Query-parser accuracy, measured separately from retrieval.

Labels (eval/queries/parser_labels.jsonl) carry their own "today" and are
parsed against a fixed vocabulary (parser_vocab.json), so this eval doesn't
depend on your library and results are reproducible.

Metrics per parser (rules / llm / auto):
  field accuracy   correct fields / union of expected and predicted fields
  field P / R      per field (a wrong extra filter hurts precision, a missed one recall)
  exact match      every filter right, nothing extra
  semantic F1      token F1 of the leftover semantic text
  latency          p50 / p95

    uv run python -m eval.parser_eval --parsers rules,llm,auto
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from api.config import ROOT
from api.search.query_parser import Vocab, clear_parse_cache, parse_query
from eval.metrics import percentile
from eval.tracking import REPORTS_DIR, git_commit, mlflow_run

LABELS = ROOT / "eval" / "queries" / "parser_labels.jsonl"
VOCAB = ROOT / "eval" / "queries" / "parser_vocab.json"
FIELDS = [
    "date_from", "date_to", "months", "years", "place", "camera", "lens", "focal_min", "focal_max",
    "aperture_min", "aperture_max", "iso_min", "iso_max", "people", "media", "orientation",
]  # fmt: skip
_STOP = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "with",
    "and",
    "photos",
    "photo",
    "pictures",
    "shots",
}


def _norm_str(s: Any) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def field_match(field: str, exp: Any, got: Any) -> bool:
    if field in ("months", "years"):
        return sorted(exp) == sorted(got)
    if field == "people":
        return sorted(p.lower() for p in exp) == sorted(p.lower() for p in got)
    if field in ("camera", "lens"):
        # both are substring filters in SQL: "X100V" and "Fujifilm X100V" select the same photos
        a, b = _norm_str(exp), _norm_str(got)
        return a in b or b in a
    if field == "place":
        return _norm_str(exp) == _norm_str(got)
    if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
        return abs(exp - got) <= max(0.06, 0.02 * abs(exp))
    return str(exp) == str(got)


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP}


def token_f1(exp: str, got: str) -> float:
    a, b = _tokens(exp), _tokens(got)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    tp = len(a & b)
    if tp == 0:
        return 0.0
    p, r = tp / len(b), tp / len(a)
    return 2 * p * r / (p + r)


def evaluate(parser: str, labels: list[dict], vocab: Vocab) -> dict[str, Any]:
    clear_parse_cache()  # measure each parser cold, not off another parser's cache
    per_field = {f: {"tp": 0, "fp": 0, "fn": 0} for f in FIELDS}
    rows, lat = [], []
    correct = union = exact = 0
    sem = []
    for lab in labels:
        today = date.fromisoformat(lab["today"])
        t0 = time.perf_counter()
        try:
            p = parse_query(lab["query"], parser, vocab=vocab, today=today)  # type: ignore[arg-type]
            got = p.filters.model_dump(mode="json", exclude_none=True)
            got = {k: v for k, v in got.items() if v not in ([], "") and k in FIELDS}
            semantic, err, source = p.semantic, p.error, str(p.source)
        except Exception as e:
            got, semantic, err, source = {}, lab["query"], str(e), "error"
        lat.append((time.perf_counter() - t0) * 1000)
        exp = {k: v for k, v in lab["filters"].items() if v not in ([], "", None)}
        ok_fields, bad = 0, []
        for f in set(exp) | set(got):
            if f in exp and f in got and field_match(f, exp[f], got[f]):
                per_field[f]["tp"] += 1
                ok_fields += 1
            else:
                if f in got:
                    per_field[f]["fp"] += 1
                if f in exp:
                    per_field[f]["fn"] += 1
                bad.append(f)
        n_union = len(set(exp) | set(got))
        correct += ok_fields
        union += n_union
        exact += not bad
        f1 = token_f1(lab["semantic"], semantic)
        sem.append(f1)
        rows.append(
            {
                "query": lab["query"],
                "expected": exp,
                "got": got,
                "semantic_expected": lab["semantic"],
                "semantic_got": semantic,
                "wrong_fields": bad,
                "semantic_f1": round(f1, 3),
                "source": source,
                "error": err,
            }
        )
    fields_out = {}
    for f, c in per_field.items():
        if c["tp"] + c["fp"] + c["fn"] == 0:
            continue
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
        rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
        fields_out[f] = {"precision": prec, "recall": rec, "support": c["tp"] + c["fn"]}
    return {
        "parser": parser,
        "n": len(labels),
        "field_accuracy": correct / union if union else 1.0,
        "exact_match": exact / len(labels),
        "semantic_f1": sum(sem) / len(sem),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "fields": fields_out,
        "rows": rows,
    }


def to_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "| parser | field acc. | exact match | semantic F1 | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['parser']} | {r['field_accuracy']:.3f} | {r['exact_match']:.3f} | {r['semantic_f1']:.3f} | "
            f"{r['latency_p50_ms']:.1f} | {r['latency_p95_ms']:.1f} |"
        )
    fields = sorted({f for r in results for f in r["fields"]})
    lines += ["", "| field | support | " + " | ".join(f"{r['parser']} P / R" for r in results) + " |",
              "|---|---:|" + "---:|" * len(results)]  # fmt: skip
    for f in fields:
        sup = max(r["fields"].get(f, {}).get("support", 0) for r in results)
        cells = []
        for r in results:
            m = r["fields"].get(f)
            cells.append(f"{m['precision']:.2f} / {m['recall']:.2f}" if m else "–")
        lines.append(f"| {f} | {sup} | " + " | ".join(cells) + " |")
    for r in results:
        wrong = [x for x in r["rows"] if x["wrong_fields"]][:8]
        if wrong:
            lines += ["", f"**{r['parser']}: sample failures**", ""]
            lines += [f"- `{x['query']}`: wrong {x['wrong_fields']}; got {x['got']}" for x in wrong]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parsers", default="rules,llm_raw,llm,hybrid,auto")
    ap.add_argument(
        "--labels", type=Path, default=LABELS, help="e.g. eval/queries/parser_labels_holdout.jsonl"
    )
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    labels = [json.loads(line) for line in a.labels.read_text().splitlines() if line.strip()]
    vocab = Vocab(**json.loads(VOCAB.read_text()))
    results = []
    for parser in a.parsers.split(","):
        r = evaluate(parser, labels, vocab)
        results.append(r)
        with mlflow_run(
            "query-parser", f"{parser}-{a.labels.stem}", enabled=not a.no_mlflow
        ) as run:
            run.log_params(
                {"parser": parser, "labels": a.labels.name, "n": r["n"], "git_commit": git_commit()}
            )
            run.log_metrics({k: v for k, v in r.items() if isinstance(v, float)})
            r["mlflow_run_id"] = run.run_id
    md = to_markdown(results)
    print(md)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out: Path = REPORTS_DIR / f"{stamp}_parser_{a.labels.stem}.json"
    out.write_text(
        json.dumps(
            {"kind": "parser", "created_at": stamp, "git_commit": git_commit(), "results": results},
            indent=2,
            default=str,
        )
        + "\n"
    )
    out.with_suffix(".md").write_text(md)
    print(f"report: {out}")


if __name__ == "__main__":
    main()
