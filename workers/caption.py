"""VLM captions, embedded for semantic search and indexed for full text.

Backends:
  florence2  microsoft Florence-2 (230M): fast, good object/scene detail. Default.
  qwen2vl    Qwen2-VL-2B-Instruct: slower, better at counts, actions and mood.
  anthropic  Claude via the API. Opt-in only: this one sends thumbnails off the
             machine, so it's for people who explicitly choose it.

Captions run on 512 px thumbnails in GPU batches and are resumable: every run
only captions photos with no caption from the current model.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import numpy as np
from PIL import Image

from api.config import get_settings
from api.db.session import get_conn
from api.ml.devices import get_device
from api.ml.text_embed import get_text_embedder
from api.paths import resolve

log = logging.getLogger(__name__)

QWEN_PROMPT = (
    "Describe this photo for a search index in 2-3 sentences: the main subjects and how many, "
    "what they are doing, the setting, time of day, lighting, weather, colours and mood. "
    "Mention any visible text. No speculation about who people are."
)


class Captioner(Protocol):
    name: str

    def caption(self, images: list[Image.Image]) -> list[str]: ...


class Florence2Captioner:
    TASK = "<MORE_DETAILED_CAPTION>"

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self.name = f"florence2:{model_id.split('/')[-1]}"
        self.device = get_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        model: Any = Florence2ForConditionalGeneration.from_pretrained(model_id, dtype=self.dtype)
        self.model = model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self._torch = torch

    def run(self, images: list[Image.Image], task: str, max_new_tokens: int = 160) -> list[str]:
        torch = self._torch
        inputs = self.processor(text=[task] * len(images), images=images, return_tensors="pt").to(
            self.device, self.dtype
        )
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, num_beams=3, do_sample=False
            )
        return [t.strip() for t in self.processor.batch_decode(out, skip_special_tokens=True)]

    def caption(self, images: list[Image.Image]) -> list[str]:
        return self.run(images, self.TASK)


class Qwen2VLCaptioner:
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.name = f"qwen2vl:{model_id.split('/')[-1]}"
        self.device = get_device()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        model: Any = Qwen2VLForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
        self.model = model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
        )
        self.processor.tokenizer.padding_side = "left"
        self._torch = torch

    def caption(self, images: list[Image.Image]) -> list[str]:
        torch = self._torch
        msgs = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QWEN_PROMPT}]}
        ]
        prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt] * len(images), images=images, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=120, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1] :]
        return [t.strip() for t in self.processor.batch_decode(out, skip_special_tokens=True)]


class AnthropicCaptioner:
    """Opt-in cloud fallback. Sends each 512 px thumbnail to the Claude API."""

    def __init__(self) -> None:
        from api.llm import _anthropic

        s = get_settings()
        self.name = f"anthropic:{s.anthropic_model}"
        self.client = _anthropic()
        self.model = s.anthropic_model

    def _one(self, im: Image.Image) -> str:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            output_config={"effort": "low"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.standard_b64encode(buf.getvalue()).decode(),
                            },
                        },
                        {"type": "text", "text": QWEN_PROMPT},
                    ],
                }
            ],
        )
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def caption(self, images: list[Image.Image]) -> list[str]:
        with ThreadPoolExecutor(4) as ex:
            return list(ex.map(self._one, images))


class FakeCaptioner:
    """Names the dominant colour. Used by tests."""

    name = "fake-captioner"

    def caption(self, images: list[Image.Image]) -> list[str]:
        from api.ml.clip import COLORS

        out = []
        for im in images:
            rgb = (
                np.asarray(im.convert("RGB").resize((4, 4)), np.float32).reshape(-1, 3).mean(0)
                / 255
            )
            name = min(COLORS, key=lambda c: float(np.sum((np.array(COLORS[c]) - rgb) ** 2)))
            out.append(f"a photo that is mostly {name}")
        return out


_captioner: Captioner | None = None
_lock = threading.Lock()


def get_captioner() -> Captioner | None:
    global _captioner
    s = get_settings()
    with _lock:
        if _captioner is None:
            if s.caption_backend == "none":
                return None
            if s.caption_backend == "florence2":
                _captioner = Florence2Captioner(s.caption_model)
            elif s.caption_backend == "qwen2vl":
                model = (
                    s.caption_model
                    if "qwen" in s.caption_model.lower()
                    else "Qwen/Qwen2-VL-2B-Instruct"
                )
                _captioner = Qwen2VLCaptioner(model)
            elif s.caption_backend == "anthropic":
                _captioner = AnthropicCaptioner()
        return _captioner


def set_captioner(c: Captioner | None) -> None:
    global _captioner
    _captioner = c


def _load(path: str) -> Image.Image | None:
    try:
        with Image.open(resolve(path) or path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def caption_pending(
    *,
    ids: Iterable[uuid.UUID] | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> int:
    cap = get_captioner()
    if cap is None:
        return 0
    s = get_settings()
    embedder = get_text_embedder()
    batch_size = batch_size or s.caption_batch_size
    extra = "AND p.id = ANY(%(ids)s)" if ids is not None else ""
    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT p.id, p.thumb_path FROM photos p
            WHERE p.media_type = 'image' AND p.thumb_path IS NOT NULL AND p.error IS NULL
              AND NOT EXISTS (SELECT 1 FROM captions c WHERE c.photo_id = p.id AND c.model = %(m)s)
              {extra}
            ORDER BY p.id LIMIT %(lim)s""",
            {"m": cap.name, "lim": limit, "ids": list(ids) if ids is not None else None},
        ).fetchall()
    done = 0
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            imgs = list(pool.map(_load, [r["thumb_path"] for r in batch]))
            ok = [(r, im) for r, im in zip(batch, imgs, strict=True) if im is not None]
            if not ok:
                continue
            try:
                texts = cap.caption([im for _, im in ok])
            except Exception as e:  # one bad batch shouldn't stop an overnight run
                log.exception("caption batch failed: %s", e)
                continue
            vecs = embedder.encode_docs(texts)
            with get_conn() as conn, conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO captions (photo_id, model, caption, embed_model, caption_embedding) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (photo_id, model) DO UPDATE SET "
                    "caption = EXCLUDED.caption, embed_model = EXCLUDED.embed_model, "
                    "caption_embedding = EXCLUDED.caption_embedding, created_at = now()",
                    [
                        (r["id"], cap.name, t, embedder.name, v)
                        for (r, _), t, v in zip(ok, texts, vecs, strict=True)
                    ],
                )
                conn.commit()
            done += len(ok)
            if progress:
                progress("caption", done, len(rows))
    return done


def reembed_captions(limit: int | None = None) -> int:
    """Re-embed existing captions after switching PS_TEXT_EMBED_MODEL."""
    embedder = get_text_embedder()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT photo_id, model, caption FROM captions WHERE embed_model IS DISTINCT FROM %s LIMIT %s",
            (embedder.name, limit),
        ).fetchall()
    for i in range(0, len(rows), 256):
        batch = rows[i : i + 256]
        vecs = embedder.encode_docs([r["caption"] for r in batch])
        with get_conn() as conn, conn.cursor() as cur:
            cur.executemany(
                "UPDATE captions SET caption_embedding = %s, embed_model = %s WHERE photo_id = %s AND model = %s",
                [
                    (v, embedder.name, r["photo_id"], r["model"])
                    for r, v in zip(batch, vecs, strict=True)
                ],
            )
            conn.commit()
    return len(rows)
