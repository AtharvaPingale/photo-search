"""The labelled query set.

eval/queries/queries.jsonl, one query per line, versioned in git:

    {"id": "q0042", "query": "coffee shop menu", "category": "text", "split": "dev",
     "relevant": ["<file_hash>", ...], "grades": {"<file_hash>": 2}, "notes": ""}

Photos are referenced by content hash, not path or database id, so labels
survive re-indexing, moving folders and rebuilding the database. The split is
assigned once from a hash of the query id and never changes; "test" is only
read by `make eval-test`, never by tuning code.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from api.config import ROOT

QUERIES_FILE = ROOT / "eval" / "queries" / "queries.jsonl"
CATEGORIES = ("objects", "attributes", "metadata", "text", "people", "hard")
Split = Literal["dev", "test"]
TEST_FRACTION = 0.3

_lock = threading.Lock()


def assign_split(qid: str) -> Split:
    h = int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 1000
    return "test" if h < TEST_FRACTION * 1000 else "dev"


class EvalQuery(BaseModel):
    id: str
    query: str
    category: str = "objects"
    split: Split = "dev"
    relevant: list[str] = Field(default_factory=list)
    grades: dict[str, int] = Field(default_factory=dict)
    notes: str = ""
    labeled_at: str | None = None

    def judgments(self) -> dict[str, int]:
        return {h: self.grades.get(h, 1) for h in self.relevant}

    @property
    def labeled(self) -> bool:
        return bool(self.relevant)


def load_queries(
    path: Path | None = None, *, split: str | None = None, labeled_only: bool = True
) -> list[EvalQuery]:
    path = path or QUERIES_FILE
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        q = EvalQuery(**json.loads(line))
        if labeled_only and not q.labeled:
            continue
        if split and split != "all" and q.split != split:
            continue
        out.append(q)
    return out


def save_queries(queries: list[EvalQuery], path: Path | None = None) -> None:
    path = path or QUERIES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(q.model_dump_json(exclude_none=True) + "\n" for q in queries))
    tmp.replace(path)


def next_id(queries: list[EvalQuery]) -> str:
    nums = [int(q.id[1:]) for q in queries if q.id[1:].isdigit()]
    return f"q{(max(nums) + 1 if nums else 1):04d}"


def upsert(q: EvalQuery, path: Path | None = None) -> EvalQuery:
    with _lock:
        qs = load_queries(path, labeled_only=False)
        if not q.id:
            q.id = next_id(qs)
            q.split = assign_split(q.id)
        else:
            # the split is fixed at creation; an edit can't move a query into dev
            existing = next((x for x in qs if x.id == q.id), None)
            q.split = existing.split if existing else assign_split(q.id)
        if q.relevant:
            q.labeled_at = datetime.now(UTC).isoformat(timespec="seconds")
        qs = [x for x in qs if x.id != q.id] + [q]
        qs.sort(key=lambda x: x.id)
        save_queries(qs, path)
    return q


def delete(qid: str, path: Path | None = None) -> bool:
    with _lock:
        qs = load_queries(path, labeled_only=False)
        keep = [x for x in qs if x.id != qid]
        if len(keep) == len(qs):
            return False
        save_queries(keep, path)
    return True
