"""Build (image, text) training pairs for fine-tuning CLIP on this library.

Sources
  caption   VLM captions, split into sentences (CLIP's text encoder sees 77 tokens)
  keywords  Lightroom / XMP keywords -> "a photo of {kw1}, {kw2}, {kw3}"
  feedback  "more like this" clicks from search: (the query typed, the photo liked)

Leakage precautions (all counted in the manifest)
  1. Every photo labelled relevant for ANY eval query (dev or test) is excluded,
     and so is every photo in the same near-duplicate or burst group, and every
     photo taken within 10 minutes of one on the same camera (the same scene shot
     twice is not an independent example).
  2. Feedback queries that match an eval query (normalised text, or token Jaccard
     >= 0.6) are dropped: training on the eval query text itself would inflate
     exactly the numbers we're trying to measure.
  3. A validation split is carved out by *day*, not by photo, so near-identical
     shots from one outing never straddle train and validation.

    uv run python -m training.build_pairs --out data/training
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

from api.config import get_settings
from api.db.session import get_conn
from api.paths import resolve
from eval.dataset import load_queries

GENERIC_KEYWORDS = {
    "photo",
    "photos",
    "image",
    "picked",
    "rejected",
    "edited",
    "export",
    "places",
    "people",
}
_WORD = re.compile(r"[a-z0-9]+")


def _norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def jaccard(a: str, b: str) -> float:
    x, y = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(x & y) / len(x | y) if x | y else 0.0


def caption_sentences(caption: str, max_sentences: int = 2) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", caption) if len(s.split()) >= 3]
    # Florence's long captions open with "The image shows ..."; drop the boilerplate
    sents = [
        re.sub(r"^(the|this) (image|photo|picture) (shows|is|depicts|features) ", "", s, flags=re.I)
        for s in sents
    ]
    return sents[:max_sentences]


def excluded_photos(conn) -> tuple[set, dict[str, int]]:
    """Eval photos + their duplicate/burst groups + same-scene neighbours."""
    queries = load_queries(split="all")
    hashes = {h for q in queries for h in q.relevant}
    rows = conn.execute(
        "SELECT id, camera, taken_at FROM photos WHERE file_hash = ANY(%s)", (list(hashes),)
    ).fetchall()
    eval_ids = {r["id"] for r in rows}
    group_ids = {
        r["photo_id"]
        for r in conn.execute(
            "SELECT g2.photo_id FROM photo_groups g1 JOIN photo_groups g2 "
            "ON g1.kind = g2.kind AND g1.group_id = g2.group_id WHERE g1.photo_id = ANY(%s)",
            (list(eval_ids),),
        )
    }
    scene_ids = set()
    for r in rows:
        if r["taken_at"] is None:
            continue
        scene_ids |= {
            x["id"]
            for x in conn.execute(
                "SELECT id FROM photos WHERE camera IS NOT DISTINCT FROM %s AND taken_at BETWEEN %s AND %s",
                (
                    r["camera"],
                    r["taken_at"] - timedelta(minutes=10),
                    r["taken_at"] + timedelta(minutes=10),
                ),
            )
        }
    excluded = eval_ids | group_ids | scene_ids
    return excluded, {
        "eval_photos": len(eval_ids),
        "dup_or_burst_neighbours": len(group_ids - eval_ids),
        "same_scene_neighbours": len(scene_ids - eval_ids - group_ids),
        "total_excluded": len(excluded),
    }


def val_by_day(taken_at, photo_id, frac: float) -> bool:
    key = taken_at.date().isoformat() if taken_at else str(photo_id)
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 1000 < frac * 1000


def build(out_dir: Path, val_frac: float = 0.05, max_keywords: int = 4) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_texts = [q.query for q in load_queries(split="all", labeled_only=False)]
    eval_norm = {_norm(t) for t in eval_texts}
    stats: Counter[str] = Counter()
    pairs: list[dict[str, Any]] = []
    with get_conn() as conn:
        excluded, leak = excluded_photos(conn)
        photos = {
            r["id"]: r
            for r in conn.execute(
                "SELECT id, file_hash, thumb_path, taken_at, keywords FROM photos "
                "WHERE media_type = 'image' AND thumb_path IS NOT NULL AND error IS NULL"
            )
        }
        for r in conn.execute("SELECT photo_id, caption FROM captions ORDER BY photo_id, model"):
            if r["photo_id"] in excluded or r["photo_id"] not in photos:
                stats["caption_excluded"] += 1
                continue
            for s in caption_sentences(r["caption"]):
                pairs.append({"photo_id": r["photo_id"], "text": s, "source": "caption"})
        for pid, p in photos.items():
            kws = [k for k in p["keywords"] if k.lower() not in GENERIC_KEYWORDS and len(k) > 1]
            if not kws:
                continue
            if pid in excluded:
                stats["keywords_excluded"] += 1
                continue
            pairs.append(
                {
                    "photo_id": pid,
                    "text": "a photo of " + ", ".join(kws[:max_keywords]),
                    "source": "keywords",
                }
            )
        for r in conn.execute("SELECT query, photo_id FROM feedback WHERE label = 1"):
            q = r["query"]
            if _norm(q) in eval_norm or any(jaccard(q, e) >= 0.6 for e in eval_texts):
                stats["feedback_dropped_eval_like"] += 1
                continue
            if r["photo_id"] in excluded or r["photo_id"] not in photos:
                stats["feedback_excluded_photo"] += 1
                continue
            pairs.append({"photo_id": r["photo_id"], "text": q, "source": "feedback"})

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for pr in pairs:
        p = photos[pr["photo_id"]]
        rec = {
            **pr,
            "photo_id": str(pr["photo_id"]),
            "file_hash": p["file_hash"],
            "image": str(resolve(p["thumb_path"])),
        }
        (val if val_by_day(p["taken_at"], pr["photo_id"], val_frac) else train).append(rec)
    for name, rows in (("train", train), ("val", val)):
        (out_dir / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    manifest = {
        "n_train": len(train),
        "n_val": len(val),
        "n_photos_train": len({x["photo_id"] for x in train}),
        "by_source": dict(Counter(x["source"] for x in train)),
        "leakage_exclusions": leak,
        "dropped": dict(stats),
        "eval_queries_seen": len(eval_texts),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=get_settings().data_dir / "training")
    ap.add_argument("--val-frac", type=float, default=0.05)
    a = ap.parse_args()
    print(json.dumps(build(a.out, a.val_frac), indent=2))


if __name__ == "__main__":
    main()
