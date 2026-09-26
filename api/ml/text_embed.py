"""Sentence embeddings for captions (384-d, bge-small by default)."""

from __future__ import annotations

import hashlib
import re
import threading

import numpy as np

from api.config import get_settings
from api.ml.devices import get_device

DIM = 384
# bge models are trained with an instruction prefix on the query side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class TextEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=get_device())
        self._lock = threading.Lock()
        dim = self.model.get_sentence_embedding_dimension()
        if dim != DIM:
            raise ValueError(f"{model_name} has dim {dim}; captions.caption_embedding is {DIM}")

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            v = self.model.encode(texts, batch_size=64, normalize_embeddings=True)
        return np.asarray(v, np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        prefix = BGE_QUERY_PREFIX if "bge" in self.name.lower() else ""
        with self._lock:
            v = self.model.encode([prefix + text], normalize_embeddings=True)
        return np.asarray(v[0], np.float32)


_WORD = re.compile(r"[a-z0-9]+")


class FakeTextEmbedder:
    """Hashed bag of words. Shared words give positive cosine similarity."""

    name = "fake-text"

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, np.float32)
        for w in _WORD.findall(text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, DIM), np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._one(text)


_embedders: dict[str, TextEmbedder | FakeTextEmbedder] = {}
_lock = threading.Lock()


def get_text_embedder(name: str | None = None) -> TextEmbedder | FakeTextEmbedder:
    name = name or get_settings().text_embed_model
    with _lock:
        if name not in _embedders:
            _embedders[name] = FakeTextEmbedder() if name == "fake-text" else TextEmbedder(name)
        return _embedders[name]
