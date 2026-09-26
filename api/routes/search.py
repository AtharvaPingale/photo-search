from __future__ import annotations

import io
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from api.search.engine import SearchRequest, SearchResponse, search
from api.search.query_parser import Mode, ParsedQuery, parse_query

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search_get(
    q: str = "",
    k: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    parse: Mode = "auto",
    model: str | None = None,
) -> SearchResponse:
    return search(SearchRequest(q=q, k=k, offset=offset, parse=parse, model=model))


@router.post("/search", response_model=SearchResponse)
def search_post(req: SearchRequest) -> SearchResponse:
    """Full request: explicit (user-edited) filters, relevance feedback
    (positive_ids / negative_ids), search-by-example (like_photo_id), signal ablation."""
    return search(req)


@router.get("/search/similar/{photo_id}", response_model=SearchResponse)
def similar(photo_id: uuid.UUID, k: Annotated[int, Query(ge=1, le=500)] = 50) -> SearchResponse:
    return search(SearchRequest(like_photo_id=photo_id, k=k))


@router.post("/search/image", response_model=SearchResponse)
def search_by_upload(
    file: Annotated[UploadFile, File()], k: Annotated[int, Query(ge=1, le=500)] = 50
) -> SearchResponse:
    """Search by an uploaded example image. The image is embedded in memory and discarded."""
    from api.ml.clip import get_clip

    data = file.file.read(40 * 1024 * 1024)
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, f"not an image: {e}") from e
    vec = get_clip().encode_images([im])[0]
    return search(SearchRequest(image_vector=vec.tolist(), k=k))


class FeedbackIn(BaseModel):
    query: str
    photo_id: uuid.UUID
    label: Literal[-1, 1]


@router.post("/feedback")
def feedback(body: FeedbackIn) -> dict[str, bool]:
    """Log a more/less-like-this click (also becomes fine-tuning data, see training/)."""
    from api.db.session import get_conn

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (query, photo_id, label) VALUES (%s, %s, %s)",
            (body.query, body.photo_id, body.label),
        )
        conn.commit()
    return {"ok": True}


@router.get("/parse", response_model=ParsedQuery)
def parse(q: str, mode: Mode = "auto") -> ParsedQuery:
    """Just the query parser: what the filter chips show before searching."""
    return parse_query(q, mode)
