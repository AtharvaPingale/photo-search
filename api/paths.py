"""Derived files (thumbnails, video frames, face crops) are stored relative to
PS_DATA_DIR, so a restored database works on a machine with a different data
directory. Originals keep absolute paths; `photo-search relink` rewrites those."""

from __future__ import annotations

from pathlib import Path

from api.config import get_settings


def stored(path: Path) -> str:
    data = get_settings().data_dir.resolve()
    p = Path(path).resolve()
    try:
        return str(p.relative_to(data))
    except ValueError:
        return str(p)


def resolve(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    p = Path(value)
    return p if p.is_absolute() else get_settings().data_dir / p
