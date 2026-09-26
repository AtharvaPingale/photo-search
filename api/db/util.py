from __future__ import annotations

from typing import Any

import numpy as np


def vec(v: Any) -> np.ndarray:
    """pgvector's psycopg loader returns `Vector` objects; everything else here wants float32 numpy."""
    if hasattr(v, "to_numpy"):
        v = v.to_numpy()
    return np.asarray(v, dtype=np.float32)
