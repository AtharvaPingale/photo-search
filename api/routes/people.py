from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.db.session import get_conn
from api.paths import resolve
from api.urls import thumb_url

router = APIRouter(tags=["people"])


class PersonUpdate(BaseModel):
    name: str | None = None
    aliases: list[str] | None = None
    hidden: bool | None = None


class MergeIn(BaseModel):
    sources: list[int]
    target: int


class SplitIn(BaseModel):
    face_ids: list[uuid.UUID] = Field(min_length=1)


class AssignIn(BaseModel):
    cluster_id: int | None


@router.get("/people")
def list_people(include_hidden: bool = False) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT pe.cluster_id, pe.name, pe.aliases, pe.hidden,
                   count(f.id) AS n_faces, count(DISTINCT f.photo_id) AS n_photos,
                   (array_agg(f.id ORDER BY f.det_score DESC))[1:6] AS sample_faces
            FROM people pe JOIN faces f ON f.cluster_id = pe.cluster_id
            WHERE %s OR NOT pe.hidden
            GROUP BY pe.cluster_id ORDER BY pe.name IS NULL, count(f.id) DESC
            """,
            (include_hidden,),
        ).fetchall()
    return [
        {**r, "sample_crops": [f"/api/faces/{fid}/crop" for fid in r.pop("sample_faces") or []]}
        for r in rows
    ]


@router.get("/people/{cluster_id}/faces")
def cluster_faces(
    cluster_id: int, limit: Annotated[int, Query(le=2000)] = 500
) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT f.id, f.photo_id, f.det_score, f.manual FROM faces f WHERE f.cluster_id = %s "
            "ORDER BY f.det_score DESC LIMIT %s",
            (cluster_id, limit),
        ).fetchall()
    return [
        {
            **r,
            "crop_url": f"/api/faces/{r['id']}/crop",
            "thumb_url": thumb_url(r["photo_id"]),
        }
        for r in rows
    ]


@router.get("/faces/unassigned")
def unassigned(limit: Annotated[int, Query(le=2000)] = 200) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, photo_id, det_score FROM faces WHERE cluster_id IS NULL ORDER BY det_score DESC LIMIT %s",
            (limit,),
        ).fetchall()
    return [{**r, "crop_url": f"/api/faces/{r['id']}/crop"} for r in rows]


@router.get("/faces/{face_id}/crop")
def crop(face_id: uuid.UUID) -> FileResponse:
    with get_conn() as conn:
        row = conn.execute("SELECT crop_path FROM faces WHERE id = %s", (face_id,)).fetchone()
    crop_path = resolve(row["crop_path"]) if row else None
    if crop_path is None or not crop_path.exists():
        raise HTTPException(404, "no crop")
    return FileResponse(
        crop_path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"}
    )


@router.put("/people/{cluster_id}")
def update_person(cluster_id: int, body: PersonUpdate) -> dict[str, bool]:
    from workers.faces import name_cluster

    name_cluster(cluster_id, body.name, body.aliases, body.hidden)
    from api.search.query_parser import clear_vocab_cache

    clear_vocab_cache()
    return {"ok": True}


@router.post("/people/merge")
def merge(body: MergeIn) -> dict[str, int]:
    from workers.faces import merge_clusters

    return {"moved_faces": merge_clusters(body.sources, body.target)}


@router.post("/people/split")
def split(body: SplitIn) -> dict[str, int]:
    from workers.faces import split_cluster

    return {"new_cluster_id": split_cluster(body.face_ids)}


@router.post("/faces/{face_id}/assign")
def assign(face_id: uuid.UUID, body: AssignIn) -> dict[str, bool]:
    from workers.faces import assign_face

    assign_face(face_id, body.cluster_id)
    return {"ok": True}
