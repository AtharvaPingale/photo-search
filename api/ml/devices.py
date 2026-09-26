from __future__ import annotations

from functools import lru_cache

from api.config import get_settings


@lru_cache
def get_device() -> str:
    """Resolve PS_DEVICE=auto to cuda / mps / cpu. GPU is used when present, CPU otherwise."""
    wanted = get_settings().device
    if wanted != "auto":
        return wanted
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
