"""Agent eval: answer correctness, groundedness, tool calls per answer.

QA set: eval/queries/agent_qa.jsonl

    {"id": "a001", "question": "What was my most photographed city in 2025?",
     "kind": "short", "expected": "Chicago"}
    {"id": "a020", "question": "Show my best sunset from each trip this year",
     "kind": "freeform", "expected": "one sunset photo per 2026 trip: Tokyo, Lisbon"}

`make_qa` generates short-answer questions whose ground truth comes from the
library itself (via the same safe query templates the agent uses, but not via
the agent), so the set adapts to any library. Add free-form ones by hand.

Scoring
  short     normalised containment of the expected answer, plus an LLM judge
  freeform  LLM judge against the expected description
  grounded  every cited photo id was returned by a tool, and the judge agrees
            the cited photos' metadata supports the answer
Judge calibration: `rate` asks you to grade a sample yourself; the report shows
judge-vs-human agreement and Cohen's kappa so the judge's numbers can be trusted
(or not).

    uv run python -m eval.agent_eval make_qa
    uv run python -m eval.agent_eval run
    uv run python -m eval.agent_eval rate --n 15
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
from eval.metrics import percentile
from eval.tracking import REPORTS_DIR, git_commit, mlflow_run

QA_FILE = ROOT / "eval" / "queries" / "agent_qa.jsonl"
RATINGS_FILE = ROOT / "eval" / "queries" / "agent_ratings.jsonl"

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "supported": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "supported", "reason"],
    "additionalProperties": False,
}
JUDGE_SYSTEM = """You grade answers from a photo-library assistant.
correct: does the ANSWER state the same fact as EXPECTED? Minor wording, extra detail and
date formats don't matter; a different place/number/date, or "couldn't find it" when EXPECTED
has an answer, is incorrect.
supported: do the CITED PHOTOS' records back up the answer's claims (dates, places, counts
plausibly)? No citations -> false, unless the answer correctly says nothing was found."""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def contains(expected: str, answer: str) -> bool:
    e, a = _norm(expected), _norm(answer)
    if not e:
        return False
    if e in a:
        return True
    # dates: accept "June 2, 2025" for "2025-06-02"
    m = re.fullmatch(r"(\d{4}) (\d{2}) (\d{2})", e)
    if m:
        d = date(int(m[1]), int(m[2]), int(m[3]))
        variants = {
            d.strftime("%B %-d %Y").lower(),
            d.strftime("%b %-d %Y").lower(),
            d.strftime("%-d %B %Y").lower(),
        }
        return any(_norm(v) in a for v in variants)
    return False


# ------------------------------------------------------------------ QA generation


def make_qa(path: Path = QA_FILE) -> list[dict[str, Any]]:
    """Short-answer questions with ground truth computed from the library."""
    from api.agent.tools import run_tool

    qa: list[dict[str, Any]] = []

    def add(question: str, expected: str | None, source: dict) -> None:
        if expected:
            qa.append({"id": f"a{len(qa) + 1:03d}", "question": question, "kind": "short",
                       "expected": expected, "source": source})  # fmt: skip

    years = [
        g["value"]
        for g in run_tool("aggregate", {"group_by": "year", "top_n": 4})[0].get("groups", [])
    ]
    for y in years:
        f = {"date_from": f"{y}-01-01", "date_to": f"{y}-12-31"}
        for gb, question in (
            ("city", f"Which city did I photograph most in {y}?"),
            ("camera", f"Which camera did I use most in {y}?"),
            ("month", f"In which month of {y} did I take the most photos?"),
            ("country", f"In {y}, which country did I take the most photos in?"),
        ):
            res, _ = run_tool("aggregate", {"group_by": gb, "filters": f, "top_n": 2})
            g = res.get("groups", [])
            # skip ties: the ground truth would be ambiguous
            if g and (len(g) == 1 or g[0]["count"] > g[1]["count"]):
                val = g[0]["value"]
                if gb == "month":
                    val = datetime.strptime(val, "%Y-%m").strftime("%B")
                add(question, str(val), {"tool": "aggregate", "group_by": gb, "filters": f})
        res, _ = run_tool(
            "metadata_query", {"template": "photos_in_range", "params": {**f, "limit": 1}}
        )
        add(
            f"How many photos did I take in {y}?",
            str(res.get("total_matching")),
            {"tool": "photos_in_range", **f},
        )
    cities = [
        g["value"]
        for g in run_tool("aggregate", {"group_by": "city", "top_n": 8})[0].get("groups", [])
    ]
    for c in cities:
        res, _ = run_tool("metadata_query", {"template": "first_and_last", "params": {"place": c}})
        if res.get("last"):
            add(
                f"When did I last take photos in {c}?",
                res["last"]["taken_at"][:10],
                {"tool": "first_and_last", "place": c},
            )
        if res.get("first"):
            add(
                f"When was my first photo in {c}?",
                res["first"]["taken_at"][:10],
                {"tool": "first_and_last", "place": c},
            )
        add(
            f"How many photos do I have from {c}?",
            str(res.get("total_matching")),
            {"tool": "first_and_last", "place": c},
        )
    people = run_tool("metadata_query", {"template": "library_overview", "params": {}})[0].get(
        "named_people", []
    )
    for p in people[:5]:
        res, _ = run_tool("metadata_query", {"template": "first_and_last", "params": {"person": p}})
        if res.get("last"):
            add(
                f"When is the most recent photo of {p}?",
                res["last"]["taken_at"][:10],
                {"tool": "first_and_last", "person": p},
            )
        res, _ = run_tool("aggregate", {"group_by": "city", "filters": {"people": [p]}, "top_n": 2})
        g = res.get("groups", [])
        if g and (len(g) == 1 or g[0]["count"] > g[1]["count"]):
            add(
                f"Where have I photographed {p} the most?",
                g[0]["value"],
                {"tool": "aggregate", "person": p},
            )
    lenses = run_tool("aggregate", {"group_by": "lens", "top_n": 2})[0].get("groups", [])
    if lenses and (len(lenses) == 1 or lenses[0]["count"] > lenses[1]["count"]):
        add(
            "Which lens do I use the most?",
            lenses[0]["value"],
            {"tool": "aggregate", "group_by": "lens"},
        )

    existing = []
    if path.exists():  # keep hand-written (freeform) questions
        existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    manual = [q for q in existing if q.get("kind") == "freeform" or q.get("manual")]
    for i, q in enumerate(manual):
        q["id"] = f"m{i + 1:03d}"
    all_q = qa[:50] + manual
    path.write_text("".join(json.dumps(q) + "\n" for q in all_q))
    return all_q


# ------------------------------------------------------------------ run


def judge(question: str, expected: str, answer: str, evidence: list[dict]) -> dict[str, Any]:
    from api.agent.tools import run_tool
    from api.llm import complete_json

    cited = []
    for ev in evidence[:6]:
        details, _ = run_tool("get_photo_details", {"photo_id": ev["photo_id"]})
        details.pop("path", None)
        cited.append(details)
    prompt = (
        f"QUESTION: {question}\nEXPECTED: {expected}\nANSWER: {answer}\n"
        f"CITED PHOTOS: {json.dumps(cited, default=str)[:6000]}"
    )
    return complete_json(JUDGE_SYSTEM, prompt, JUDGE_SCHEMA, timeout=120, max_tokens=500)


def run(
    path: Path = QA_FILE, limit: int | None = None, use_judge: bool = True, mlflow: bool = True
) -> dict[str, Any]:
    from api.agent.graph import run_agent
    from api.llm import model_label

    qa = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    qa = [q for q in qa if q.get("expected")][
        :limit
    ]  # hand-written stubs need an expected answer first
    rows = []
    for q in qa:
        t0 = time.perf_counter()
        ans = run_agent(q["question"])
        row: dict[str, Any] = {
            "id": q["id"], "question": q["question"], "kind": q["kind"], "expected": q["expected"],
            "answer": ans.answer, "evidence": [e.model_dump() for e in ans.evidence],
            "invalid_citations": ans.invalid_citations, "tool_calls": ans.tool_calls,
            "steps": [s.model_dump() for s in ans.steps], "error": ans.error,
            "latency_s": round(time.perf_counter() - t0, 2),
        }  # fmt: skip
        row["contains"] = contains(q["expected"], ans.answer) if q["kind"] == "short" else None
        if use_judge and not ans.error:
            try:
                row["judge"] = judge(q["question"], q["expected"], ans.answer, row["evidence"])
            except Exception as e:
                row["judge"] = {"error": str(e)}
        j = row.get("judge") or {}
        row["correct"] = (
            bool(row["contains"])
            if q["kind"] == "short" and row["contains"]
            else bool(j.get("correct"))
        )
        row["grounded"] = ans.grounded and bool(j.get("supported", True))
        rows.append(row)
        print(f"{q['id']} {'OK ' if row['correct'] else 'BAD'} {'G' if row['grounded'] else '-'} "
              f"{ans.tool_calls} calls  {q['question'][:60]} -> {ans.answer[:80]}")  # fmt: skip

    n = len(rows) or 1
    report = {
        "kind": "agent",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "llm": model_label(),
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / n,
        "accuracy_short": _mean([r["correct"] for r in rows if r["kind"] == "short"]),
        "accuracy_freeform": _mean([r["correct"] for r in rows if r["kind"] == "freeform"]),
        "groundedness": sum(r["grounded"] for r in rows) / n,
        "hallucinated_citation_rate": sum(bool(r["invalid_citations"]) for r in rows) / n,
        "tool_calls_mean": sum(r["tool_calls"] for r in rows) / n,
        "latency_p50_s": percentile([r["latency_s"] for r in rows], 50),
        "latency_p95_s": percentile([r["latency_s"] for r in rows], 95),
        "errors": sum(bool(r["error"]) for r in rows),
        "judge_calibration": calibration(rows),
        "rows": rows,
    }
    with mlflow_run("agent", report["llm"], enabled=mlflow) as mr:
        mr.log_params({"llm": report["llm"], "n": report["n"], "git_commit": report["git_commit"]})
        mr.log_metrics({k: v for k, v in report.items() if isinstance(v, float)})
        report["mlflow_run_id"] = mr.run_id
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = REPORTS_DIR / f"{stamp}_agent.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    (REPORTS_DIR / f"{stamp}_agent.md").write_text(to_markdown(report))
    report["report_path"] = str(out)
    return report


def _mean(xs: list[bool]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def calibration(rows: list[dict]) -> dict[str, Any]:
    """Judge vs. your own ratings (agent_ratings.jsonl) on the same answers."""
    if not RATINGS_FILE.exists():
        return {"n": 0}
    human = {}
    for line in RATINGS_FILE.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            human[(r["id"], _norm(r["answer"]))] = r["correct"]
    pairs = []
    for r in rows:
        key = (r["id"], _norm(r["answer"]))
        if key in human and isinstance(r.get("judge"), dict) and "correct" in r["judge"]:
            pairs.append((human[key], bool(r["judge"]["correct"])))
    if not pairs:
        return {"n": 0}
    agree = sum(a == b for a, b in pairs) / len(pairs)
    ph = sum(a for a, _ in pairs) / len(pairs)
    pj = sum(b for _, b in pairs) / len(pairs)
    pe = ph * pj + (1 - ph) * (1 - pj)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": len(pairs), "agreement": agree, "cohens_kappa": kappa}


def rate(n: int = 15) -> None:
    """Grade the latest run's answers yourself, to calibrate the judge."""
    latest = max(REPORTS_DIR.glob("*_agent.json"), default=None)
    if latest is None:
        raise SystemExit("run the agent eval first")
    rows = json.loads(latest.read_text())["rows"][:n]
    with RATINGS_FILE.open("a") as f:
        for r in rows:
            print(f"\nQ: {r['question']}\nexpected: {r['expected']}\nanswer:   {r['answer']}")
            v = input("correct? [y/n/s=skip] ").strip().lower()
            if v in ("y", "n"):
                f.write(
                    json.dumps({"id": r["id"], "answer": r["answer"], "correct": v == "y"}) + "\n"
                )


def to_markdown(r: dict[str, Any]) -> str:
    cal = r["judge_calibration"]
    lines = [
        f"# Agent eval ({r['llm']}, {r['n']} questions, commit `{r['git_commit']}`)",
        "",
        "| accuracy | short | freeform | grounded | hallucinated citations | tool calls / answer | p50 s | p95 s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {r['accuracy']:.2f} | {r['accuracy_short']:.2f} | {r['accuracy_freeform']:.2f} | {r['groundedness']:.2f} | "
        f"{r['hallucinated_citation_rate']:.2f} | {r['tool_calls_mean']:.1f} | {r['latency_p50_s']:.1f} | {r['latency_p95_s']:.1f} |",
        "",
        f"Judge calibration: {cal.get('n', 0)} human-rated answers"
        + (
            f", agreement {cal['agreement']:.2f}, Cohen's kappa {cal['cohens_kappa']:.2f}"
            if cal.get("n")
            else ""
        ),
        "",
        "## Failures",
        "",
    ]
    for row in [x for x in r["rows"] if not x["correct"]][:15]:
        why = (row.get("judge") or {}).get("reason", "")
        lines.append(
            f"- **{row['question']}** expected `{row['expected']}`, got: {row['answer'][:200]} ({why[:160]})"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument("--qa-file", type=Path, default=QA_FILE)
    sub.add_parser("make_qa")
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int)
    r.add_argument("--no-judge", action="store_true")
    r.add_argument("--no-mlflow", action="store_true")
    rt = sub.add_parser("rate")
    rt.add_argument("--n", type=int, default=15)
    a = ap.parse_args()
    if a.cmd == "make_qa":
        qa = make_qa(a.qa_file)
        print(f"wrote {len(qa)} questions to {a.qa_file}")
    elif a.cmd == "run":
        rep = run(a.qa_file, limit=a.limit, use_judge=not a.no_judge, mlflow=not a.no_mlflow)
        print(to_markdown(rep))
    else:
        rate(a.n)


if __name__ == "__main__":
    main()
