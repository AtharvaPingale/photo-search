"""Test fixtures.

Tests run against a real Postgres + pgvector (an embedded server from
`pgserver`, or PS_TEST_DATABASE_URL if set) with deterministic fake models,
so they need no GPU, no network and no Docker. Tests marked `slow` use the
real OpenCLIP / text models.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="photo-search-test-"))
os.environ.update(
    PS_DATA_DIR=str(_TMP / "data"),
    PS_IMAGE_MODEL="fake-clip",
    PS_TEXT_EMBED_MODEL="fake-text",
    PS_LLM_PROVIDER="none",
    PS_CAPTION_BACKEND="none",
    PS_OCR_BACKEND="none",
    PS_DEVICE="cpu",
    PS_FUSION_WEIGHTS_FILE=str(_TMP / "no-weights.json"),
    LANGSMITH_TRACING="false",
)

from PIL import Image  # noqa: E402
from PIL.TiffImagePlugin import IFDRational  # noqa: E402

TABLES = [
    "feedback", "imports",
    "album_photos", "albums", "photo_groups", "people", "faces", "ocr_text", "captions",
    "image_embeddings", "photos",
]  # fmt: skip


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    url = os.environ.get("PS_TEST_DATABASE_URL")
    srv = None
    if not url:
        import pgserver

        srv = pgserver.get_server(_TMP / "pg", cleanup_mode="stop")
        url = srv.get_uri()
    os.environ["PS_DATABASE_URL"] = url
    from api.config import get_settings

    get_settings.cache_clear()
    from api.db.migrate import migrate

    migrate(url)
    yield url
    from api.db.session import close_pool

    close_pool()
    if srv is not None:
        srv.cleanup()
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture
def db(pg_url: str):
    """A clean database for each test."""
    from api.db.session import get_conn
    from api.search import engine, query_parser

    with get_conn() as conn:
        conn.execute("TRUNCATE " + ", ".join(TABLES) + " CASCADE")
        conn.commit()
    engine.reset_caches()
    query_parser.clear_vocab_cache()
    yield
    engine.reset_caches()
    query_parser.clear_vocab_cache()


@pytest.fixture(autouse=True)
def _tiny_geocoder():
    from workers.geocode import ReverseGeocoder, set_geocoder

    set_geocoder(
        ReverseGeocoder(
            names=["Chicago", "Evanston", "Tokyo", "Paris", "Columbus"],
            admin1=["Illinois", "Illinois", "Tokyo", "Île-de-France", "Ohio"],
            country=["United States", "United States", "Japan", "France", "United States"],
            lat=[41.85, 42.04, 35.69, 48.85, 39.96],
            lon=[-87.65, -87.69, 139.69, 2.35, -83.00],
        )
    )
    yield
    set_geocoder(None)


def _dms(x: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    x = abs(x)
    d = int(x)
    m = int((x - d) * 60)
    s = round(((x - d) * 60 - m) * 60 * 100)
    return IFDRational(d, 1), IFDRational(m, 1), IFDRational(s, 100)


def make_photo(
    path: Path,
    color: tuple[int, int, int] = (200, 30, 30),
    *,
    size: tuple[int, int] = (320, 240),
    taken: datetime | None = datetime(2025, 6, 1, 18, 30),
    camera: tuple[str, str] | None = ("Canon", "Canon EOS R6"),
    lens: str | None = "RF50mm F1.8 STM",
    focal: float | None = 50.0,
    aperture: float | None = 1.8,
    iso: int | None = 400,
    gps: tuple[float, float] | None = None,
    pattern: bool = False,
) -> Path:
    """A JPEG with realistic EXIF. `pattern` draws a stripe so near-duplicates differ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    if pattern:
        for x in range(0, size[0], 40):
            for y in range(size[1]):
                im.putpixel((x, y), (255 - color[0], 255 - color[1], 255 - color[2]))
    exif = Image.Exif()
    if camera:
        exif[0x010F], exif[0x0110] = camera
    e = exif.get_ifd(0x8769)
    if taken:
        e[0x9003] = taken.strftime("%Y:%m:%d %H:%M:%S")
    if lens:
        e[0xA434] = lens
    if focal:
        e[0x920A] = IFDRational(int(focal * 10), 10)
    if aperture:
        e[0x829D] = IFDRational(int(aperture * 10), 10)
    if iso:
        e[0x8827] = iso
    e[0x829A] = IFDRational(1, 250)
    if gps:
        g = exif.get_ifd(0x8825)
        g[1] = "N" if gps[0] >= 0 else "S"
        g[2] = _dms(gps[0])
        g[3] = "E" if gps[1] >= 0 else "W"
        g[4] = _dms(gps[1])
    im.save(path, "JPEG", exif=exif, quality=92)
    return path


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A small library: colours, cameras, places and dates chosen so tests can
    assert on search results."""
    root = tmp_path / "library"
    make_photo(root / "2025/chicago/red_car.jpg", (220, 20, 20), gps=(41.88, -87.63))
    make_photo(root / "2025/chicago/red_sign.jpg", (200, 40, 30), gps=(41.90, -87.62),
               taken=datetime(2025, 6, 2, 12, 0))  # fmt: skip
    make_photo(root / "2025/tokyo/blue_night.jpg", (20, 20, 200), gps=(35.68, 139.70),
               taken=datetime(2025, 7, 10, 22, 0), camera=("FUJIFILM", "X100V"),
               lens=None, focal=23.0, aperture=2.0, iso=3200)  # fmt: skip
    make_photo(root / "2024/paris/green_park.jpg", (30, 180, 40), gps=(48.86, 2.34),
               taken=datetime(2024, 8, 15, 10, 0), lens="FE 85mm F1.8", focal=85.0,
               camera=("SONY", "ILCE-7M3"))  # fmt: skip
    make_photo(root / "2024/misc/yellow_flower.jpg", (230, 220, 30), taken=datetime(2024, 4, 3, 9, 0),
               lens="EF-S 18-150mm", focal=150.0)  # fmt: skip
    make_photo(
        root / "2023/white_wall.jpg", (245, 245, 245), taken=datetime(2023, 12, 25, 11, 0), gps=None
    )
    return root


@pytest.fixture
def indexed(db, library: Path) -> Path:
    from workers.embed import embed_pending
    from workers.ingest import scan

    scan([library])
    embed_pending()
    return library
