"""Captions, OCR, fusion across signals, faces, organisation."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from api.config import get_settings
from api.db.session import get_conn
from api.search.engine import SearchRequest, search
from api.search.filters import SearchFilters
from tests.conftest import make_photo


def ids_by_name() -> dict[str, object]:
    with get_conn() as conn:
        return {Path(r["path"]).name: r["id"] for r in conn.execute("SELECT id, path FROM photos")}


def names(resp) -> list[str]:
    return [Path(h.path).name for h in resp.hits]


@pytest.fixture
def settings(monkeypatch):
    """Override settings for one test: settings(face_roots='["/x"]', ...)."""

    def apply(**kw: str) -> None:
        for k, v in kw.items():
            monkeypatch.setenv(f"PS_{k.upper()}", v)
        get_settings.cache_clear()

    yield apply
    monkeypatch.undo()
    get_settings.cache_clear()


# ------------------------------------------------------------------ captions + OCR + fusion


class FakeOCR:
    name = "fake-ocr"

    def __init__(self, texts: dict[tuple[int, int, int], str]):
        self.texts = texts

    def read(self, rgb: np.ndarray) -> str:
        c = tuple(int(x) for x in rgb.reshape(-1, 3).mean(0).round())
        best = min(self.texts, key=lambda k: sum((a - b) ** 2 for a, b in zip(k, c, strict=True)))
        return self.texts[best]


def test_captions_ocr_and_fusion(indexed, settings):
    from api.search import engine
    from workers import caption, ocr

    caption.set_captioner(caption.FakeCaptioner())
    ocr.set_engine(FakeOCR({(220, 20, 20): "", (200, 40, 30): "STOP", (20, 20, 200): "RAMEN MENU"}))
    settings(ocr_backend="paddleocr")
    try:
        assert caption.caption_pending() == 6
        assert caption.caption_pending() == 0  # resumable
        stats = ocr.ocr_pending(threshold=0.0)  # gate open: OCR everything
        assert stats["ocr_run"] == 6
        assert ocr.ocr_pending(threshold=0.0)["ocr_run"] == 0
    finally:
        caption.set_captioner(None)
        ocr.set_engine(None)
    engine.reset_caches()

    # OCR-only query: the menu text is only in the blue photo's OCR
    resp = search(SearchRequest(q="menu", parse="off", signals=["ocr"]))
    assert names(resp) == ["blue_night.jpg"]
    # caption keyword signal: "mostly yellow"
    resp = search(SearchRequest(q="yellow", parse="off", signals=["keyword"]))
    assert names(resp)[0] == "yellow_flower.jpg"
    # fusion: all signals contribute and are reported
    resp = search(SearchRequest(q="ramen menu", parse="off"))
    assert set(resp.signals) == {"clip", "caption", "keyword", "ocr"}
    assert names(resp)[0] == "blue_night.jpg"
    assert resp.hits[0].ranks.get("ocr") == 1


def test_ocr_gate_skips_low_scores(indexed, settings):
    from workers import ocr

    ocr.set_engine(FakeOCR({(0, 0, 0): "x"}))
    settings(ocr_backend="paddleocr")
    try:
        stats = ocr.ocr_pending(threshold=1.01)  # nothing passes
    finally:
        ocr.set_engine(None)
    assert stats == {"gated_out": 6, "ocr_run": 0, "with_text": 0}
    with get_conn() as conn:
        assert (
            conn.execute("SELECT count(*) n FROM ocr_text WHERE NOT ran_ocr").fetchone()["n"] == 6
        )
    assert ocr.regate(-1.0) == 6


# ------------------------------------------------------------------ faces


class FakeFaces:
    """One 'face' per photo, whose identity is the photo's dominant colour channel."""

    def detect(self, rgb: np.ndarray):
        from workers.faces import DetectedFace

        m = rgb.reshape(-1, 3).mean(0)
        if m.max() - m.min() < 50:  # white wall: no face
            return []
        e = np.zeros(512, np.float32)
        e[int(np.argmax(m))] = 1.0
        e[10 + int(m.sum()) % 7] = 0.05  # small per-photo jitter
        e /= np.linalg.norm(e)
        return [DetectedFace((10, 10, 90, 90), 0.99, e)]


def test_faces_are_opt_in(indexed, settings):
    from workers import faces

    faces.set_engine(FakeFaces())
    try:
        assert faces.detect_pending() == 0  # no PS_FACE_ROOTS -> nothing
        settings(face_roots=f'["{indexed / "2025"}"]')
        assert faces.detect_pending() == 3  # only the 2025 folder
    finally:
        faces.set_engine(None)


def test_face_clustering_naming_and_people_queries(indexed, settings, tmp_path):
    from workers import faces

    # add more "people" photos so clusters reach min size
    for i in range(3):
        make_photo(indexed / f"more/red_{i}.jpg", (210 + i, 30, 20), taken=datetime(2025, 6, 3 + i))
        make_photo(
            indexed / f"more/blue_{i}.jpg", (20, 30, 210 + i), taken=datetime(2025, 7, 3 + i)
        )
    from workers.embed import embed_pending
    from workers.ingest import scan

    scan([indexed])
    embed_pending()
    settings(face_roots=f'["{indexed}"]')
    faces.set_engine(FakeFaces())
    try:
        faces.detect_pending()
    finally:
        faces.set_engine(None)
    out = faces.cluster_faces(min_cluster_size=3, min_samples=1)
    assert out["clusters"] >= 2
    with get_conn() as conn:
        by_photo = {
            Path(r["path"]).name: r["cluster_id"]
            for r in conn.execute(
                "SELECT p.path, f.cluster_id FROM faces f JOIN photos p ON p.id = f.photo_id"
            )
        }
    red_c, blue_c = by_photo["red_car.jpg"], by_photo["blue_night.jpg"]
    assert red_c is not None and blue_c is not None and red_c != blue_c
    assert by_photo["red_0.jpg"] == red_c

    faces.name_cluster(red_c, "Rohan")
    faces.name_cluster(blue_c, "Atharva", ["me"])
    from api.search import query_parser

    query_parser.clear_vocab_cache()
    resp = search(SearchRequest(q="Rohan", parse="rules"))
    assert resp.parsed.filters.people == ["Rohan"]
    assert all(by_photo[n] == red_c for n in names(resp))
    resp = search(SearchRequest(q="me at night", parse="rules"))
    assert resp.parsed.filters.people == ["me"]
    assert set(names(resp)) == {n for n, c in by_photo.items() if c == blue_c}

    # names survive re-clustering
    faces.cluster_faces(min_cluster_size=3, min_samples=1)
    with get_conn() as conn:
        names_now = {
            r["name"] for r in conn.execute("SELECT name FROM people WHERE name IS NOT NULL")
        }
    assert names_now == {"Rohan", "Atharva"}

    # split one face out, then merge it back
    with get_conn() as conn:
        fid = conn.execute(
            "SELECT id FROM faces WHERE cluster_id = %s LIMIT 1", (red_c,)
        ).fetchone()["id"]
    new_c = faces.split_cluster([fid])
    assert new_c not in (red_c, blue_c)
    assert faces.merge_clusters([new_c], red_c) == 1
    # manual edits survive reclustering
    faces.cluster_faces(min_cluster_size=3, min_samples=1)
    with get_conn() as conn:
        row = conn.execute("SELECT cluster_id, manual FROM faces WHERE id = %s", (fid,)).fetchone()
    assert row["cluster_id"] == red_c and row["manual"]

    # the wipe removes everything, crops included
    assert Path(get_settings().faces_dir).exists()
    out = faces.wipe_faces()
    assert out["faces_deleted"] > 0
    with get_conn() as conn:
        assert conn.execute("SELECT count(*) n FROM faces").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) n FROM people").fetchone()["n"] == 0
    assert not Path(get_settings().faces_dir).exists()


# ------------------------------------------------------------------ organisation


def test_duplicates_bursts_albums(indexed):
    from workers import organize
    from workers.embed import embed_pending
    from workers.ingest import scan

    # exact copy at a different size + a burst of three near-identical frames
    im = Image.open(indexed / "2025/tokyo/blue_night.jpg")
    im.resize((160, 120)).save(indexed / "copy_small.jpg", quality=80)
    base = datetime(2025, 7, 10, 22, 0, 0)
    for i in range(3):
        make_photo(indexed / f"burst/b{i}.jpg", (30 + i, 160, 60), taken=base + timedelta(seconds=i),
                   camera=("SONY", "ILCE-7M3"), pattern=True)  # fmt: skip
    scan([indexed])
    embed_pending()

    d = organize.find_duplicates()
    assert d["groups"] >= 1
    b = organize.find_bursts()
    assert b["groups"] == 1 and b["photos"] == 3
    with get_conn() as conn:
        dup_names = {
            Path(r["path"]).name
            for r in conn.execute(
                "SELECT p.path FROM photo_groups g JOIN photos p ON p.id = g.photo_id WHERE g.kind = 'duplicate' "
                "AND g.group_id = (SELECT group_id FROM photo_groups gg JOIN photos pp ON pp.id = gg.photo_id "
                "WHERE gg.kind = 'duplicate' AND pp.path LIKE '%%copy_small.jpg')"
            )
        }
        best = conn.execute(
            "SELECT p.path FROM photo_groups g JOIN photos p ON p.id = g.photo_id WHERE g.kind='duplicate' AND g.is_best "
            "AND p.path LIKE '%%blue_night.jpg'"
        ).fetchone()
    assert {"copy_small.jpg", "blue_night.jpg"} <= dup_names
    assert best is not None  # the larger original is the suggested keeper

    a = organize.build_albums(min_photos=2, gap_h=24, llm_titles=False)
    assert a["albums"] >= 1
    with get_conn() as conn:
        titles = [r["title"] for r in conn.execute("SELECT title FROM albums ORDER BY start_at")]
    assert any("Chicago" in t for t in titles)

    # album_id filter works in search
    with get_conn() as conn:
        aid = conn.execute("SELECT id FROM albums WHERE title LIKE 'Chicago%%'").fetchone()["id"]
    resp = search(SearchRequest(q="red", parse="off", filters=SearchFilters(album_id=str(aid))))
    assert set(names(resp)) == {"red_car.jpg", "red_sign.jpg"}


def test_segmentation_and_trips():
    from workers.organize import group_trips, home_region, segment_timeline

    def p(day: int, hour: int, admin1: str, country: str = "United States"):
        return {"taken_at": datetime(2025, 1, day, hour), "place_name": admin1, "admin1": admin1,
                "country": country, "id": f"{day}-{hour}"}  # fmt: skip

    photos = [p(d, 12, "Ohio") for d in range(1, 10)]  # home
    photos += [p(12, 9, "Tokyo", "Japan"), p(12, 20, "Tokyo", "Japan"), p(13, 10, "Kyoto", "Japan")]
    photos.sort(key=lambda x: x["taken_at"])
    segs = segment_timeline(photos, gap_h=8)
    assert home_region(photos) == "Ohio|United States"
    trips = group_trips(segs, home_region(photos))
    japan = [t for t in trips if t.photos[0]["country"] == "Japan"]
    # Tokyo day 12 (two segments) + Kyoto day 13 merge into one trip: same country, <48h gaps,
    # but Kyoto is a different region, so it only merges if regions match
    assert sum(len(t.photos) for t in japan) == 3


def test_video_frames_are_searchable(db, tmp_path):
    """Scene sampling on a real (generated) video: red then blue."""
    import subprocess

    from workers.embed import embed_pending
    from workers.ingest import scan
    from workers.video import ffmpeg_exe, index_videos

    root = tmp_path / "vids"
    root.mkdir()
    out = root / "clip.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=red:s=160x120:d=2",
         "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=2",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
         "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )  # fmt: skip
    stats = scan([root])
    assert stats.new == 1
    n = index_videos()
    assert n >= 2
    embed_pending()
    resp = search(SearchRequest(q="blue", parse="off", k=5))
    top = resp.hits[0]
    assert top.is_video_frame and top.path.endswith("clip.mp4")
    assert top.frame_ts is not None and top.frame_ts >= 1.5  # the blue half
    assert len({h.video_id for h in resp.hits}) == len(resp.hits)  # one hit per video
    # the moved-video path rewrite keeps frames attached
    shutil.move(out, root / "renamed.mp4")
    assert scan([root]).moved == 1
    with get_conn() as conn:
        paths = [r["path"] for r in conn.execute("SELECT path FROM photos WHERE is_video_frame")]
    assert paths and all(p.startswith(str(root / "renamed.mp4") + "#t=") for p in paths)
    _ = os
