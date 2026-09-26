"""Phase 9 (phone access: auth, fast media) and Phase 10 (import, backups, relink/repair, watch)."""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.config import get_settings
from api.db.session import get_conn
from tests.conftest import make_photo


@pytest.fixture
def settings(monkeypatch):
    def apply(**kw: str) -> None:
        for k, v in kw.items():
            monkeypatch.setenv(f"PS_{k.upper()}", v)
        get_settings.cache_clear()

    yield apply
    monkeypatch.undo()
    get_settings.cache_clear()


def _client():
    from api.main import create_app

    return TestClient(create_app())


def _first_id() -> str:
    with get_conn() as conn:
        return str(conn.execute("SELECT id FROM photos ORDER BY path LIMIT 1").fetchone()["id"])


# ------------------------------------------------------------------ auth


def test_auth_token_and_cookie(indexed, settings):
    settings(auth_token="s3cret-token")
    c = _client()
    assert c.get("/api/health").status_code == 200  # always open
    assert c.get("/api/auth/status").json() == {"required": True, "authenticated": False}
    assert c.get("/api/search", params={"q": "red"}).status_code == 401
    pid = _first_id()
    assert c.get(f"/api/photos/{pid}/thumb").status_code == 401

    # bearer header works for scripts
    r = c.get(
        "/api/search",
        params={"q": "red", "parse": "off"},
        headers={"Authorization": "Bearer s3cret-token"},
    )
    assert r.status_code == 200

    # wrong token is rejected; right token sets an HttpOnly cookie that <img> requests carry
    assert c.post("/api/auth/login", json={"token": "nope"}).status_code == 401
    r = c.post("/api/auth/login", json={"token": "s3cret-token"})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "s3cret-token" not in cookie
    assert c.get(f"/api/photos/{pid}/thumb").status_code == 200
    assert c.get("/api/auth/status").json()["authenticated"]
    c.post("/api/auth/logout")
    c.cookies.clear()
    assert c.get(f"/api/photos/{pid}/thumb").status_code == 401


def test_login_rate_limited(db, settings):
    from api import auth

    auth._failures.clear()
    settings(auth_token="tok")
    c = _client()
    codes = [c.post("/api/auth/login", json={"token": "bad"}).status_code for _ in range(12)]
    assert codes[:10] == [401] * 10 and codes[-1] == 429
    auth._failures.clear()


def test_no_auth_when_token_unset(indexed):
    assert _client().get("/api/search", params={"q": "red", "parse": "off"}).status_code == 200


# ------------------------------------------------------------------ fast media for phones


def test_small_thumbs_display_and_download(indexed):
    c = _client()
    hit = c.get("/api/search", params={"q": "red", "k": 1, "parse": "off"}).json()["hits"][0]
    assert "size=256" in hit["thumb_url"] and "v=" in hit["thumb_url"]

    r = c.get(hit["thumb_url"])
    assert r.headers["content-type"] == "image/webp"
    assert "immutable" in r.headers["cache-control"]
    assert len(r.content) < 15_000

    r = c.get(hit["display_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert c.get(hit["display_url"]).content == r.content  # cached rendition

    r = c.get(f"/api/photos/{hit['photo_id']}/original", params={"download": 1})
    assert r.headers["content-disposition"].startswith("attachment")
    assert Path(hit["path"]).name in r.headers["content-disposition"]

    big = c.get("/api/search", params={"q": "red", "k": 50, "parse": "off"})
    assert big.headers.get("content-encoding") == "gzip"


# ------------------------------------------------------------------ auto import


def test_import_from_backup_folder(db, tmp_path, settings):
    from workers.importer import import_and_index

    backup = tmp_path / "phone-backup"
    lib = tmp_path / "library"
    lib.mkdir()
    settings(import_dirs=f'["{backup}"]', import_dest=str(lib), photo_roots=f'["{lib}"]')
    a = make_photo(backup / "IMG_0001.JPG", (200, 20, 20), taken=datetime(2025, 3, 4, 10, 0))
    make_photo(backup / "IMG_0002.JPG", (20, 20, 200), taken=datetime(2024, 12, 31, 23, 0))
    (backup / "IMG_0001.xmp").write_text("<x:xmpmeta/>")
    shutil.copy(a, backup / "IMG_0001 (1).JPG")  # a backup app's duplicate upload
    (backup / "IMG_0003.JPG.part").write_bytes(b"half a file")
    for p in backup.iterdir():
        os.utime(p, (time.time() - 60, time.time() - 60))

    out = import_and_index()
    assert out["import"] == {
        "imported": 2,
        "duplicates": 0,
        "already_seen": 1,
        "not_ready": 0,
        "errors": 0,
    }
    assert (lib / "2025/2025-03/IMG_0001.JPG").exists()
    assert (lib / "2025/2025-03/IMG_0001.xmp").exists()
    assert (lib / "2024/2024-12/IMG_0002.JPG").exists()
    assert out["scan"]["new"] == 2
    with get_conn() as conn:
        assert (
            conn.execute("SELECT count(*) n FROM image_embeddings").fetchone()["n"] == 2
        )  # indexed

    # deleting a photo from the library doesn't bring it back on the next sync
    (lib / "2024/2024-12/IMG_0002.JPG").unlink()
    again = import_and_index()
    assert again["import"]["imported"] == 0 and again["import"]["already_seen"] == 3
    assert not (lib / "2024/2024-12/IMG_0002.JPG").exists()


def test_import_skips_files_still_uploading(db, tmp_path, settings):
    from workers.importer import import_new

    backup, lib = tmp_path / "b", tmp_path / "lib"
    lib.mkdir()
    settings(import_dirs=f'["{backup}"]', import_dest=str(lib))
    make_photo(backup / "fresh.jpg")  # mtime = now
    assert import_new(wait_s=30).not_ready == 1
    assert import_new(wait_s=0).summary()["imported"] == 1


def test_import_name_collision_keeps_both(db, tmp_path, settings):
    from workers.importer import import_new

    backup, lib = tmp_path / "b", tmp_path / "lib"
    lib.mkdir()
    settings(import_dest=str(lib))
    make_photo(lib / "2025/2025-06/IMG_1.jpg", (1, 2, 3))
    make_photo(backup / "IMG_1.jpg", (200, 100, 50))
    st = import_new([backup], wait_s=0)
    assert [p.name for p in st.imported] == ["IMG_1-1.jpg"]


# ------------------------------------------------------------------ backups, relink, repair


def _pg_bin() -> str:
    import pgserver

    return str(Path(pgserver.__file__).parent / "pginstall" / "bin")


def test_backup_and_restore_roundtrip(indexed, tmp_path, monkeypatch, pg_url):
    from api.db import backup as bk
    from api.db.session import connect

    monkeypatch.setenv("PS_PG_BIN_DIR", _pg_bin())
    dump = bk.backup(pg_url, tmp_path / "backups")
    assert (
        dump.exists()
        and dump.suffix == ".dump"
        and not list((tmp_path / "backups").glob("*.partial"))
    )

    # restore into a separate, empty database
    with connect(pg_url, vector=False) as conn:
        conn.autocommit = True
        conn.execute("DROP DATABASE IF EXISTS restore_check")
        conn.execute("CREATE DATABASE restore_check")
    target = pg_url.replace("/postgres?", "/restore_check?")
    bk.restore(dump, target)
    with connect(target) as conn:
        assert conn.execute("SELECT count(*) n FROM photos").fetchone()["n"] == 6
        assert conn.execute("SELECT count(*) n FROM image_embeddings").fetchone()["n"] == 6
    with connect(pg_url, vector=False) as conn:
        conn.autocommit = True
        conn.execute("DROP DATABASE restore_check WITH (FORCE)")

    # prune keeps the newest dump no matter how old
    old = tmp_path / "backups" / "photos-old.dump"
    old.write_bytes(b"x")
    os.utime(old, (0, 0))
    os.utime(dump, (1, 1))
    assert bk.prune(tmp_path / "backups", keep_days=1) == [old]
    assert dump.exists()


def test_relink_and_repair(indexed, tmp_path):
    from workers.maintenance import relink, repair

    moved = tmp_path / "moved-library"
    shutil.move(indexed, moved)
    assert relink(str(indexed), str(moved), dry_run=True) == 6
    assert relink(str(indexed), str(moved)) == 6
    with get_conn() as conn:
        paths = [r["path"] for r in conn.execute("SELECT path FROM photos")]
    assert all(p.startswith(str(moved)) and Path(p).exists() for p in paths)

    shutil.rmtree(get_settings().thumbs_dir)
    out = repair()
    assert out["thumbnails_rebuilt"] == 6 and out["originals_missing"] == 0
    c = _client()
    assert c.get(f"/api/photos/{_first_id()}/thumb").status_code == 200


def test_watcher_batch_indexes_moves_and_deletes(indexed):
    from workers.watch import process

    new = make_photo(indexed / "new/fresh.jpg", (10, 200, 10))
    out = process({str(new)}, set(), [], inline=True)
    assert out["indexed"] == 1
    with get_conn() as conn:
        n = conn.execute(
            "SELECT count(*) n FROM image_embeddings e JOIN photos p ON p.id = e.photo_id WHERE p.path = %s",
            (str(new),),
        ).fetchone()["n"]
    assert n == 1  # searchable right away

    dest = indexed / "new/renamed.jpg"
    shutil.move(new, dest)
    assert process(set(), set(), [(str(new), str(dest))], inline=True)["moved"] == 1
    dest.unlink()
    assert process(set(), {str(dest)}, [], inline=True)["removed"] == 1
