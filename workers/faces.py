"""Faces: detection + ArcFace embeddings (InsightFace buffalo_l), HDBSCAN clustering,
naming, merge/split, and a one-command wipe.

Privacy rules, enforced here rather than in the UI:
  * faces are only computed for photos under PS_FACE_ROOTS (opt-in per folder);
  * everything face-related lives in `faces`, `people` and data/faces/, and
    `photo-search faces wipe` removes all three.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from api.config import get_settings
from api.db.session import get_conn
from api.db.util import vec
from api.paths import resolve, stored

log = logging.getLogger(__name__)

MIN_DET_SCORE = 0.65
MIN_FACE_PX = 36
DETECT_SIDE = 1600


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]
    score: float
    embedding: np.ndarray  # 512-d, L2-normalised


class FaceEngine(Protocol):
    def detect(self, rgb: np.ndarray) -> list[DetectedFace]: ...


class InsightFaceEngine:
    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        from api.ml.devices import get_device

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if get_device() == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=os.environ.get("INSIGHTFACE_HOME", str(Path.home() / ".insightface")),
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self.app.prepare(ctx_id=0 if get_device() == "cuda" else -1, det_size=(640, 640))

    def detect(self, rgb: np.ndarray) -> list[DetectedFace]:
        out = []
        for f in self.app.get(np.ascontiguousarray(rgb[:, :, ::-1])):  # insightface wants BGR
            x0, y0, x1, y1 = (round(v) for v in f.bbox)
            out.append(
                DetectedFace(
                    (x0, y0, x1, y1), float(f.det_score), np.asarray(f.normed_embedding, np.float32)
                )
            )
        return out


_engine: FaceEngine | None = None
_lock = threading.Lock()


def get_engine() -> FaceEngine:
    global _engine
    with _lock:
        if _engine is None:
            _engine = InsightFaceEngine()
        return _engine


def set_engine(e: FaceEngine | None) -> None:
    global _engine
    _engine = e


def _opted_in_sql(roots: list[Path]) -> tuple[str, dict[str, Any]]:
    if not roots:
        return "FALSE", {}
    clauses = []
    params: dict[str, Any] = {}
    for i, r in enumerate(roots):
        clauses.append(f"(p.path = %(fr{i})s OR p.path LIKE %(frl{i})s)")
        root = str(Path(r).expanduser().resolve())
        params[f"fr{i}"] = root
        params[f"frl{i}"] = (
            root.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%"
        )
    return "(" + " OR ".join(clauses) + ")", params


def detect_pending(
    *,
    ids: Iterable[uuid.UUID] | None = None,
    limit: int | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> int:
    from workers.media import open_image

    s = get_settings()
    opted, params = _opted_in_sql(s.face_roots)
    if opted == "FALSE":
        log.info("no PS_FACE_ROOTS configured: face detection is opt-in, nothing to do")
        return 0
    extra = "AND p.id = ANY(%(ids)s)" if ids is not None else ""
    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT p.id, p.path, p.thumb_path, p.is_video_frame FROM photos p
            WHERE p.media_type = 'image' AND p.error IS NULL AND p.faces_indexed_at IS NULL
              AND {opted} {extra}
            ORDER BY p.id LIMIT %(lim)s""",
            {**params, "lim": limit, "ids": list(ids) if ids is not None else None},
        ).fetchall()
    engine = get_engine()
    s.faces_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        faces: list[DetectedFace] = []
        try:
            src = (resolve(r["thumb_path"]) if r["is_video_frame"] else None) or Path(r["path"])
            im = open_image(src, max_side=DETECT_SIDE)
            im.thumbnail((DETECT_SIDE, DETECT_SIDE))
            rgb = np.asarray(im)
            faces = [
                f for f in engine.detect(rgb)
                if f.score >= MIN_DET_SCORE and min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= MIN_FACE_PX
            ]  # fmt: skip
        except Exception as e:
            log.warning("face detection failed for %s: %s", r["path"], e)
        with get_conn() as conn:
            conn.execute("DELETE FROM faces WHERE photo_id = %s", (r["id"],))
            for f in faces:
                fid = uuid.uuid4()
                crop = _save_crop(im, f.bbox, fid)
                conn.execute(
                    "INSERT INTO faces (id, photo_id, bbox, det_score, embedding, crop_path) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (fid, r["id"], list(f.bbox), f.score, f.embedding, stored(crop)),
                )
            conn.execute("UPDATE photos SET faces_indexed_at = now() WHERE id = %s", (r["id"],))
            conn.commit()
        if progress:
            progress("faces", i + 1, len(rows))
    return len(rows)


def _save_crop(im: Image.Image, bbox: tuple[int, int, int, int], fid: uuid.UUID) -> Path:
    x0, y0, x1, y1 = bbox
    pad = int(0.25 * max(x1 - x0, y1 - y0))
    box = (max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad))
    crop = im.crop(box)
    crop.thumbnail((160, 160))
    path = get_settings().faces_dir / str(fid)[:2] / f"{fid}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path, "JPEG", quality=85)
    return path


# ------------------------------------------------------------------ clustering


def cluster_faces(min_cluster_size: int = 4, min_samples: int | None = 3) -> dict[str, int]:
    """HDBSCAN over ArcFace embeddings. Manual assignments are fixed points; new
    clusters inherit the id (and so the name) of the old cluster they overlap most."""
    from sklearn.cluster import HDBSCAN

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, embedding, cluster_id, manual FROM faces ORDER BY id"
        ).fetchall()
    if not rows:
        return {"faces": 0, "clusters": 0, "noise": 0}
    ids = [r["id"] for r in rows]
    X = np.stack([vec(r["embedding"]) for r in rows])
    old = {r["id"]: r["cluster_id"] for r in rows}
    manual = {r["id"] for r in rows if r["manual"]}

    if len(rows) >= max(min_cluster_size, 2):
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
            cluster_selection_method="leaf",
        ).fit_predict(X)  # fmt: skip
    else:
        labels = np.full(len(rows), -1)

    # map each new label to an old cluster id by majority overlap (greedy, largest first)
    groups: dict[int, list[uuid.UUID]] = {}
    for fid, lab in zip(ids, labels, strict=True):
        if lab >= 0:
            groups.setdefault(int(lab), []).append(fid)
    used: set[int] = set()
    next_id = max([c for c in old.values() if c is not None] + [-1]) + 1
    new_assign: dict[uuid.UUID, int | None] = {fid: None for fid in ids}
    for _, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        votes = Counter(old[m] for m in members if old[m] is not None)
        target = next(
            (c for c, n in votes.most_common() if c not in used and n * 2 >= len(members)), None
        )
        if target is None:
            target = next_id
            next_id += 1
        used.add(target)
        for m in members:
            new_assign[m] = target
    for fid in manual:  # hand edits always win
        new_assign[fid] = old[fid]

    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE faces SET cluster_id = %s WHERE id = %s",
            [(c, fid) for fid, c in new_assign.items() if c != old[fid]],
        )
        live = {c for c in new_assign.values() if c is not None}
        cur.executemany(
            "INSERT INTO people (cluster_id) VALUES (%s) ON CONFLICT DO NOTHING",
            [(c,) for c in live],
        )
        # drop unnamed, now-empty clusters; keep named ones (a name is user data)
        cur.execute(
            "DELETE FROM people pe WHERE pe.name IS NULL AND NOT EXISTS "
            "(SELECT 1 FROM faces f WHERE f.cluster_id = pe.cluster_id)"
        )
        conn.commit()
    return {
        "faces": len(rows),
        "clusters": len(live),
        "noise": sum(1 for c in new_assign.values() if c is None),
    }


def name_cluster(
    cluster_id: int, name: str | None, aliases: list[str] | None = None, hidden: bool | None = None
) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO people (cluster_id, name, aliases, hidden) VALUES (%s, %s, %s, coalesce(%s, FALSE)) "
            "ON CONFLICT (cluster_id) DO UPDATE SET name = EXCLUDED.name, "
            "aliases = CASE WHEN %s THEN EXCLUDED.aliases ELSE people.aliases END, "
            "hidden = coalesce(%s, people.hidden)",
            (cluster_id, name or None, aliases or [], hidden, aliases is not None, hidden),
        )
        conn.commit()


def merge_clusters(sources: list[int], target: int) -> int:
    with get_conn() as conn:
        n = conn.execute(
            "UPDATE faces SET cluster_id = %s, manual = TRUE WHERE cluster_id = ANY(%s)",
            (target, [s for s in sources if s != target]),
        ).rowcount
        # target keeps its name; otherwise inherit the first named source
        conn.execute(
            "INSERT INTO people (cluster_id) VALUES (%s) ON CONFLICT DO NOTHING", (target,)
        )
        conn.execute(
            """UPDATE people t SET name = coalesce(t.name, s.name),
                   aliases = coalesce((SELECT array_agg(DISTINCT a) FROM unnest(t.aliases || s.aliases) a), '{}')
               FROM (SELECT name, aliases FROM people WHERE cluster_id = ANY(%s)
                     ORDER BY name NULLS LAST LIMIT 1) s
               WHERE t.cluster_id = %s""",
            (sources, target),
        )
        conn.execute(
            "DELETE FROM people WHERE cluster_id = ANY(%s) AND cluster_id <> %s", (sources, target)
        )
        conn.commit()
    return n


def split_cluster(face_ids: list[uuid.UUID]) -> int:
    """Move the given faces into a brand-new cluster; returns its id."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT greatest(coalesce(max(f.cluster_id), -1), coalesce((SELECT max(cluster_id) FROM people), -1)) + 1 "
            "AS n FROM faces f"
        ).fetchone()
        new_id = int(row["n"]) if row else 0
        conn.execute("INSERT INTO people (cluster_id) VALUES (%s)", (new_id,))
        conn.execute(
            "UPDATE faces SET cluster_id = %s, manual = TRUE WHERE id = ANY(%s)", (new_id, face_ids)
        )
        conn.commit()
    return new_id


def assign_face(face_id: uuid.UUID, cluster_id: int | None) -> None:
    """Move one face to a cluster, or detach it ("not this person") with None."""
    with get_conn() as conn:
        if cluster_id is not None:
            conn.execute(
                "INSERT INTO people (cluster_id) VALUES (%s) ON CONFLICT DO NOTHING", (cluster_id,)
            )
        conn.execute(
            "UPDATE faces SET cluster_id = %s, manual = TRUE WHERE id = %s", (cluster_id, face_id)
        )
        conn.commit()


def wipe_faces() -> dict[str, int]:
    """Delete every face detection, embedding, cluster, name and crop."""
    with get_conn() as conn:
        n_faces = conn.execute("DELETE FROM faces").rowcount
        n_people = conn.execute("DELETE FROM people").rowcount
        conn.execute("UPDATE photos SET faces_indexed_at = NULL WHERE faces_indexed_at IS NOT NULL")
        conn.commit()
    shutil.rmtree(get_settings().faces_dir, ignore_errors=True)
    return {"faces_deleted": n_faces, "people_deleted": n_people}
