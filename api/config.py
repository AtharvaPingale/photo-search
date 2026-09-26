"""Runtime configuration, read from environment variables (prefix PS_) or .env.

Everything defaults to a single-machine, local-only setup: no setting here
sends photos anywhere. The only outbound calls are model downloads on first
use and, if you opt in, an API-hosted LLM for query parsing / the agent
(which only ever sees text: queries, captions, metadata).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PS_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # storage
    database_url: str = "postgresql://photos:photos@localhost:5432/photos"
    redis_url: str = "redis://localhost:6379/0"
    data_dir: Path = ROOT / "data"
    backups_dir: Path = ROOT / "backups"
    backup_keep_days: int = 14

    # library
    photo_roots: list[Path] = Field(default_factory=list)
    # Face processing is opt-in per folder. Empty means faces are never computed.
    face_roots: list[Path] = Field(default_factory=list)
    video_enabled: bool = True
    # Camera-roll backup folders (PhotoSync, Syncthing, a synced iCloud/Google Photos
    # folder...). New files there are copied into import_dest as YYYY/YYYY-MM/name.
    import_dirs: list[Path] = Field(default_factory=list)
    import_dest: Path | None = None  # default: the first photo root
    import_mode: Literal["copy", "move"] = "copy"

    # Remote access. When set, every API call needs this token (see api/auth.py).
    auth_token: str | None = None

    # models
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    image_model: str = "openclip-vitb32"
    text_embed_model: str = "BAAI/bge-small-en-v1.5"
    caption_backend: Literal["florence2", "qwen2vl", "anthropic", "none"] = "florence2"
    caption_model: str = "florence-community/Florence-2-base"
    ocr_backend: Literal["paddleocr", "florence2", "none"] = "paddleocr"
    # CLIP-probe threshold for "this photo probably contains text". Lower runs OCR on more photos.
    ocr_gate_threshold: float = 0.5
    embed_batch_size: int = 64
    caption_batch_size: int = 16

    # LLM (query parsing, album titles, agent). "ollama" keeps everything local.
    llm_provider: Literal["ollama", "anthropic", "none"] = "ollama"
    llm_model: str = "qwen2.5:14b-instruct"
    ollama_url: str = "http://localhost:11434"
    anthropic_model: str = "claude-opus-5"
    llm_timeout_s: float = 20.0

    # search
    default_k: int = 50
    hnsw_ef_search: int = 100
    rrf_k: int = 60
    # Fusion weights per signal; tuned on the dev split by `make tune-fusion`.
    fusion_weights: dict[str, float] = Field(
        default_factory=lambda: {"clip": 1.0, "caption": 0.6, "keyword": 0.5, "ocr": 0.8}
    )
    fusion_weights_file: Path = ROOT / "eval" / "fusion_weights.json"

    # thumbnails
    thumb_size: int = 512

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def small_thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs_256"

    @property
    def display_dir(self) -> Path:
        return self.data_dir / "display"

    @property
    def faces_dir(self) -> Path:
        return self.data_dir / "faces"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def geonames_dir(self) -> Path:
        return self.data_dir / "geonames"


@lru_cache
def get_settings() -> Settings:
    return Settings()
