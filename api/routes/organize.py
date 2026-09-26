from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query

from api.db.session import get_conn
from api.urls import thumb_url

router = APIRouter(tags=["organize"])


@router.get("/groups/{kind}")
def groups(
    kind: Literal["duplicate", "burst"],
    limit: Annotated[int, Query(le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[dict[str, Any]]:
    """Duplicate or burst groups, largest first; each photo flagged if it's the suggested pick."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            WITH g AS (
                SELECT group_id, count(*) AS n FROM photo_groups WHERE kind = %(k)s
                GROUP BY group_id ORDER BY count(*) DESC, group_id LIMIT %(lim)s OFFSET %(off)s
            )
            SELECT g.group_id, g.n, pg.photo_id, pg.is_best, pg.score, p.path, p.file_hash, p.taken_at, p.width,
                   p.height, p.sharpness
            FROM g JOIN photo_groups pg ON pg.kind = %(k)s AND pg.group_id = g.group_id
            JOIN photos p ON p.id = pg.photo_id
            ORDER BY g.n DESC, g.group_id, p.taken_at, p.id
            """,
            {"k": kind, "lim": limit, "off": offset},
        ).fetchall()
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        grp = out.setdefault(
            r["group_id"], {"group_id": r["group_id"], "size": r["n"], "photos": []}
        )
        grp["photos"].append(
            {
                "photo_id": r["photo_id"], "is_best": r["is_best"], "path": r["path"],
                "taken_at": r["taken_at"], "width": r["width"], "height": r["height"],
                "sharpness": r["sharpness"], "thumb_url": thumb_url(r["photo_id"], r["file_hash"]),
            }
        )  # fmt: skip
    return list(out.values())
