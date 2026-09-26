"""URLs for photo media. The content hash is part of the URL, so responses can be
cached as immutable: a phone re-opening the grid never re-downloads a thumbnail,
and a photo whose pixels change gets a new URL."""

from __future__ import annotations

import uuid


def thumb_url(photo_id: uuid.UUID | str, file_hash: str | None = None, size: int = 256) -> str:
    v = f"&v={file_hash[:12]}" if file_hash else ""
    return f"/api/photos/{photo_id}/thumb?size={size}{v}"


def display_url(photo_id: uuid.UUID | str, file_hash: str | None = None) -> str:
    return f"/api/photos/{photo_id}/display" + (f"?v={file_hash[:12]}" if file_hash else "")


def original_url(photo_id: uuid.UUID | str) -> str:
    return f"/api/photos/{photo_id}/original"
