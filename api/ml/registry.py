"""Image-text model registry.

A model name is what gets stored in image_embeddings.model, so eval runs can
compare "openclip-vitb32" against "openclip-vitl14" or "finetuned-v1" without
re-indexing anything. Fine-tuned models live in data/models/<name>/spec.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from api.config import get_settings


@dataclass(frozen=True)
class ImageModelSpec:
    name: str
    arch: str
    pretrained: str | None
    dim: int
    checkpoint: Path | None = None  # fine-tuned state_dict (merged weights)
    fake: bool = False  # deterministic stand-in for tests, no download

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "arch": self.arch,
            "pretrained": self.pretrained,
            "dim": self.dim,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
        }


BUILTIN: dict[str, ImageModelSpec] = {
    "openclip-vitb32": ImageModelSpec("openclip-vitb32", "ViT-B-32", "laion2b_s34b_b79k", 512),
    "openclip-vitb16": ImageModelSpec("openclip-vitb16", "ViT-B-16", "laion2b_s34b_b88k", 512),
    "openclip-vitl14": ImageModelSpec("openclip-vitl14", "ViT-L-14", "laion2b_s32b_b82k", 768),
    "fake-clip": ImageModelSpec("fake-clip", "fake", None, 512, fake=True),
}


def models_dir() -> Path:
    return get_settings().models_dir


def get_image_model_spec(name: str | None = None) -> ImageModelSpec:
    name = name or get_settings().image_model
    if name in BUILTIN:
        return BUILTIN[name]
    spec_path = models_dir() / name / "spec.json"
    if spec_path.exists():
        raw = json.loads(spec_path.read_text())
        ckpt = raw.get("checkpoint")
        return ImageModelSpec(
            name=name,
            arch=raw["arch"],
            pretrained=raw.get("pretrained"),
            dim=int(raw["dim"]),
            checkpoint=(spec_path.parent / ckpt) if ckpt else None,
        )
    raise KeyError(f"unknown image model {name!r}; known: {sorted(list_image_models())}")


def list_image_models() -> list[str]:
    names = list(BUILTIN)
    d = models_dir()
    if d.exists():
        names += [p.parent.name for p in d.glob("*/spec.json")]
    return names


def save_finetuned_spec(name: str, base: ImageModelSpec, checkpoint_file: str) -> Path:
    out = models_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    spec = {
        "name": name,
        "arch": base.arch,
        "pretrained": base.pretrained,
        "dim": base.dim,
        "checkpoint": checkpoint_file,
        "base": base.name,
    }
    path = out / "spec.json"
    path.write_text(json.dumps(spec, indent=2))
    return path
