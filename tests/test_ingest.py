from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from api.db.session import get_conn
from api.paths import resolve
from tests.conftest import make_photo
from workers import media
from workers.ingest import scan


def rows() -> dict[str, dict]:
    with get_conn() as conn:
        return {Path(r["path"]).name: r for r in conn.execute("SELECT * FROM photos").fetchall()}


def test_exif_extraction(tmp_path: Path):
    p = make_photo(tmp_path / "a.jpg", gps=(41.8781, -87.6298))
    ex = media.read_exif(p)
    assert ex.taken_at == datetime(2025, 6, 1, 18, 30, tzinfo=UTC)
    assert ex.camera == "Canon EOS R6"
    assert ex.lens == "RF50mm F1.8 STM"
    assert ex.focal_length == 50.0
    assert ex.aperture == 1.8
    assert ex.iso == 400
    assert ex.shutter == "1/250"
    assert abs(ex.lat - 41.8781) < 1e-3 and abs(ex.lon + 87.6298) < 1e-3


def test_camera_name_normalisation():
    assert media.camera_name("Canon", "Canon EOS R6") == "Canon EOS R6"
    assert media.camera_name("NIKON CORPORATION", "NIKON Z 6") == "NIKON Z 6"
    assert media.camera_name("FUJIFILM", "X100V") == "Fujifilm X100V"
    assert media.camera_name("Apple", "iPhone 15 Pro") == "Apple iPhone 15 Pro"
    assert media.camera_name(None, "X100V") == "X100V"


def test_datetime_and_offset():
    dt, off = media.parse_exif_datetime("2024:03:05 07:08:09", "-05:00")
    assert dt == datetime(2024, 3, 5, 7, 8, 9, tzinfo=UTC) and off == -300
    assert media.parse_exif_datetime("0000:00:00 00:00:00") == (None, None)


def test_xmp_keywords(tmp_path: Path):
    xmp = """<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description>
      <dc:subject><rdf:Bag><rdf:li>beach</rdf:li><rdf:li>sunset</rdf:li></rdf:Bag></dc:subject>
      <lr:hierarchicalSubject><rdf:Bag><rdf:li>Places|USA|Chicago</rdf:li></rdf:Bag></lr:hierarchicalSubject>
      </rdf:Description></rdf:RDF></x:xmpmeta>"""
    p = make_photo(tmp_path / "k.jpg")
    p.with_suffix(".xmp").write_text(xmp)
    assert media.read_keywords(p) == ["beach", "sunset", "Places", "USA", "Chicago"]


def test_scan_extracts_metadata_and_places(db, library: Path):
    stats = scan([library])
    assert stats.new == 6 and stats.errors == 0
    r = rows()
    assert r["red_car.jpg"]["place_name"] == "Chicago"
    assert r["red_car.jpg"]["country"] == "United States"
    assert r["blue_night.jpg"]["place_name"] == "Tokyo"
    assert r["blue_night.jpg"]["camera"] == "Fujifilm X100V"
    assert r["green_park.jpg"]["lens"] == "FE 85mm F1.8"
    assert r["white_wall.jpg"]["lat"] is None
    for row in r.values():
        assert not Path(row["thumb_path"]).is_absolute()  # portable across data dirs
        assert resolve(row["thumb_path"]).exists()
        assert row["phash"] and row["sharpness"] is not None
        assert row["width"] == 320 and row["height"] == 240


def test_rescan_is_incremental(db, library: Path):
    scan([library])
    again = scan([library])
    assert again.new == again.changed == again.moved == 0
    assert again.unchanged == 6


def test_rescan_detects_change_move_touch_delete(db, library: Path):
    scan([library])
    before = rows()

    # content change: same path, new pixels
    make_photo(library / "2023/white_wall.jpg", (10, 10, 10))
    # move: same bytes, new path -> keeps id
    shutil.move(library / "2024/misc/yellow_flower.jpg", library / "2024/yellow_flower.jpg")
    # touch: mtime changes, bytes don't
    p = library / "2025/chicago/red_sign.jpg"
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 100))
    # delete
    (library / "2024/paris/green_park.jpg").unlink()

    stats = scan([library])
    assert (stats.changed, stats.moved, stats.touched, stats.deleted) == (1, 1, 1, 1)
    after = rows()
    assert "green_park.jpg" not in after
    assert after["yellow_flower.jpg"]["id"] == before["yellow_flower.jpg"]["id"]
    assert after["yellow_flower.jpg"]["path"].endswith("2024/yellow_flower.jpg")
    assert after["white_wall.jpg"]["id"] == before["white_wall.jpg"]["id"]
    assert after["white_wall.jpg"]["file_hash"] != before["white_wall.jpg"]["file_hash"]


def test_change_clears_derived_data(db, library: Path):
    from workers.embed import embed_pending

    scan([library])
    assert embed_pending() == 6
    make_photo(library / "2023/white_wall.jpg", (10, 10, 10))
    scan([library])
    assert embed_pending() == 1  # only the changed photo is re-embedded


def test_corrupt_file_recorded_not_fatal(db, library: Path):
    (library / "broken.jpg").write_bytes(b"not a jpeg")
    stats = scan([library])
    assert stats.errors == 1 and stats.new == 7
    assert rows()["broken.jpg"]["error"]


def test_geocoder_prefers_the_city_over_its_neighbourhoods():
    from workers.geocode import ReverseGeocoder

    g = ReverseGeocoder(
        names=["Tokyo", "Hatsudai", "Paris", "Paris 15 Vaugirard", "Chicago", "Evanston"],
        admin1=[""] * 6,
        country=[""] * 6,
        lat=[35.6895, 35.678, 48.8534, 48.8412, 41.85, 42.0411],
        lon=[139.6917, 139.686, 2.3488, 2.3003, -87.65, -87.6901],
        population=[8_336_599, 8_629, 2_138_551, 229_713, 2_720_546, 75_000],
    )
    assert g.lookup(35.678, 139.686).name == "Tokyo"  # a neighbourhood entry is absorbed
    assert g.lookup(48.8412, 2.3003).name == "Paris"  # so is a named district, however big
    assert g.lookup(42.045, -87.688).name == "Evanston"  # a real town beside Chicago is not
    assert g.lookup(0.0, 0.0) is None  # nothing within 150 km
