"""OCR, gated so it only runs where there's likely text.

Stage 1 (free): a zero-shot CLIP probe on the embedding we already stored
("a sign with writing" vs "a landscape"...). Photos below the threshold get an
ocr_text row with ran_ocr = FALSE and are never OCR'd.
Stage 2: PaddleOCR (its own DB text detector, then recognition) on a
1600 px rendition of the original, since text needs more pixels than CLIP does.

On a typical personal library the gate skips ~85-90% of photos; `photo-search
ocr --threshold` trades recall on text queries against OCR time, and the eval's
"text" category measures what that costs.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import numpy as np

from api.config import get_settings
from api.db.session import get_conn
from api.db.util import vec
from api.ml.clip import get_clip, text_likelihood
from api.paths import resolve

log = logging.getLogger(__name__)
MIN_SCORE = 0.6
OCR_SIDE = 1600


class OcrEngine(Protocol):
    name: str

    def read(self, rgb: np.ndarray) -> str: ...


class PaddleEngine:
    name = "paddleocr"

    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )

    def read(self, rgb: np.ndarray) -> str:
        res = self.ocr.predict(np.ascontiguousarray(rgb[:, :, ::-1]))  # PaddleOCR expects BGR
        if not res:
            return ""
        page = res[0]
        lines = [
            t
            for t, s in zip(page["rec_texts"], page["rec_scores"], strict=False)
            if s >= MIN_SCORE and t.strip()
        ]
        return "\n".join(lines)


class FlorenceEngine:
    name = "florence2-ocr"

    def __init__(self) -> None:
        from workers.caption import Florence2Captioner

        self.f = Florence2Captioner(get_settings().caption_model)

    def read(self, rgb: np.ndarray) -> str:
        from PIL import Image

        return self.f.run([Image.fromarray(rgb)], "<OCR>", max_new_tokens=256)[0]


_engine: OcrEngine | None = None
_lock = threading.Lock()


def get_engine() -> OcrEngine | None:
    global _engine
    backend = get_settings().ocr_backend
    with _lock:
        if _engine is None and backend != "none":
            _engine = PaddleEngine() if backend == "paddleocr" else FlorenceEngine()
        return _engine


def set_engine(e: OcrEngine | None) -> None:
    global _engine
    _engine = e


def ocr_pending(
    *,
    ids: Iterable[uuid.UUID] | None = None,
    limit: int | None = None,
    threshold: float | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, int]:
    from workers.media import open_image

    s = get_settings()
    threshold = s.ocr_gate_threshold if threshold is None else threshold
    engine = get_engine()
    if engine is None:
        return {"skipped_backend_none": 1}
    model = get_clip().spec.name
    extra = "AND p.id = ANY(%(ids)s)" if ids is not None else ""
    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT p.id, p.path, p.is_video_frame, p.thumb_path, e.embedding FROM photos p
            JOIN image_embeddings e ON e.photo_id = p.id AND e.model = %(m)s
            WHERE NOT EXISTS (SELECT 1 FROM ocr_text o WHERE o.photo_id = p.id) {extra}
            ORDER BY p.id LIMIT %(lim)s""",
            {"m": model, "lim": limit, "ids": list(ids) if ids is not None else None},
        ).fetchall()
    if not rows:
        return {"gated_out": 0, "ocr_run": 0, "with_text": 0}
    gate = text_likelihood(model, np.stack([vec(r["embedding"]) for r in rows]))
    stats = {"gated_out": 0, "ocr_run": 0, "with_text": 0}
    for i, (r, g) in enumerate(zip(rows, gate, strict=True)):
        text, ran = "", False
        if g >= threshold:
            try:
                # video frames only exist as their extracted still
                src = (resolve(r["thumb_path"]) if r["is_video_frame"] else None) or Path(r["path"])
                im = open_image(src, max_side=OCR_SIDE)
                im.thumbnail((OCR_SIDE, OCR_SIDE))
                text = engine.read(np.asarray(im))
                ran = True
                stats["ocr_run"] += 1
                stats["with_text"] += bool(text)
            except Exception as e:
                log.warning("OCR failed for %s: %s", r["path"], e)
        else:
            stats["gated_out"] += 1
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO ocr_text (photo_id, engine, gate_score, ran_ocr, text) VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT (photo_id) DO UPDATE SET engine = EXCLUDED.engine, gate_score = EXCLUDED.gate_score, "
                "ran_ocr = EXCLUDED.ran_ocr, text = EXCLUDED.text, created_at = now()",
                (r["id"], engine.name, float(g), ran, text),
            )
            conn.commit()
        if progress:
            progress("ocr", i + 1, len(rows))
    return stats


def regate(threshold: float) -> int:
    """After lowering the threshold: forget gated-out rows that now pass, so they get OCR'd."""
    with get_conn() as conn:
        n = conn.execute(
            "DELETE FROM ocr_text WHERE NOT ran_ocr AND gate_score >= %s", (threshold,)
        ).rowcount
        conn.commit()
    return n
