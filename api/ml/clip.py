"""CLIP image/text encoders (OpenCLIP), plus a deterministic fake for tests.

All outputs are L2-normalised float32 numpy arrays so cosine similarity is a
dot product everywhere (numpy and pgvector's <=>).
"""

from __future__ import annotations

import hashlib
import re
import threading
from functools import lru_cache
from typing import Protocol

import numpy as np
from PIL import Image

from api.ml.devices import get_device
from api.ml.registry import ImageModelSpec, get_image_model_spec

# Prompts for the zero-shot "does this photo contain readable text?" probe that
# gates OCR. It reuses the image embedding we already have, so it costs one
# dot product per photo.
TEXT_PROMPTS = [
    "a photo of a sign with writing on it",
    "a photo of a menu",
    "a photo of a document or page of text",
    "a photo of a storefront with a name on it",
    "a screenshot with text",
    "a poster with words",
]
NO_TEXT_PROMPTS = [
    "a photo of a landscape",
    "a photo of a person",
    "a photo of an animal",
    "a photo of food",
    "a photo of a room",
    "a photo of a city street",
]


class ClipEncoder(Protocol):
    spec: ImageModelSpec

    @property
    def dim(self) -> int: ...

    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class OpenClipEncoder:
    def __init__(self, spec: ImageModelSpec, device: str | None = None):
        import open_clip
        import torch

        self.spec = spec
        self.device = device or get_device()
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec.arch, pretrained=spec.pretrained, device="cpu"
        )
        if spec.checkpoint:
            state = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(spec.arch)
        self._lock = threading.Lock()
        self._torch = torch

    @property
    def dim(self) -> int:
        return self.spec.dim

    def _autocast(self):
        torch = self._torch
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type="cpu", enabled=False)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), np.float32)
        torch = self._torch
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        with self._lock, torch.inference_mode(), self._autocast():
            feats = self.model.encode_image(batch)
        return _normalize(feats.float().cpu().numpy())

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        torch = self._torch
        tokens = self.tokenizer(texts).to(self.device)
        with self._lock, torch.inference_mode(), self._autocast():
            feats = self.model.encode_text(tokens)
        return _normalize(feats.float().cpu().numpy())


# --------------------------------------------------------------------------
# Fake encoder: colour-aware and deterministic, so tests can assert that
# "red" retrieves the red photos without downloading a 600 MB model.

COLORS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
    "orange": (1.0, 0.5, 0.0),
    "purple": (0.5, 0.0, 0.5),
}
_WORD = re.compile(r"[a-z]+")


def _color_feat(rgb: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    r, g, b = (float(c) for c in rgb)
    # centred so black and white are distinguishable under cosine similarity
    return np.array([r - 0.5, g - 0.5, b - 0.5, (r + g + b) / 3 - 0.5], np.float32)


class FakeClipEncoder:
    COLOR_W = 8.0

    def __init__(self, spec: ImageModelSpec):
        self.spec = spec

    @property
    def dim(self) -> int:
        return self.spec.dim

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        out = np.zeros((len(images), self.dim), np.float32)
        for i, im in enumerate(images):
            small = np.asarray(im.convert("RGB").resize((8, 8)), np.float32) / 255.0
            out[i, :4] = _color_feat(small.reshape(-1, 3).mean(0)) * self.COLOR_W
            gray = small.mean(-1).reshape(-1)
            out[i, 8:72] = gray - gray.mean()  # spatial layout: near-dupes stay near
        out[:, 4] = 0.05  # keeps all-black images from being the zero vector
        return _normalize(out)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            words = _WORD.findall(t.lower())
            cols = [COLORS[w] for w in words if w in COLORS]
            if cols:
                out[i, :4] = _color_feat(np.mean(cols, 0)) * self.COLOR_W
            for w in words:
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, 128 + h % (self.dim - 128)] += 1.0
            out[i, 4] = 0.05
        return _normalize(out)


_encoders: dict[str, ClipEncoder] = {}
_enc_lock = threading.Lock()


def get_clip(name: str | None = None) -> ClipEncoder:
    spec = get_image_model_spec(name)
    with _enc_lock:
        enc = _encoders.get(spec.name)
        if enc is None:
            enc = FakeClipEncoder(spec) if spec.fake else OpenClipEncoder(spec)
            _encoders[spec.name] = enc
        return enc


@lru_cache(maxsize=8)
def _probe_matrix(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    enc = get_clip(model_name)
    return enc.encode_texts(TEXT_PROMPTS), enc.encode_texts(NO_TEXT_PROMPTS)


def text_likelihood(model_name: str, image_embs: np.ndarray) -> np.ndarray:
    """P(photo contains readable text) from a zero-shot CLIP probe, in [0, 1]."""
    pos, neg = _probe_matrix(model_name)
    logits = 100.0 * image_embs @ np.concatenate([pos, neg]).T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p[:, : len(pos)].sum(axis=1)
