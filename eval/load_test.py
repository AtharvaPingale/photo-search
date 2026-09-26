"""Scale test: index build time, query latency and storage at 100k+ photos.

Synthetic rows (random unit-norm embeddings, plausible metadata) go into a
*separate* Postgres schema so the real library is untouched. Query latency is
measured through the real search engine: CLIP text encoding + the same SQL,
with and without metadata filters.

    uv run python -m eval.load_test --n 100000
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from api.config import get_settings
from eval.metrics import percentile
from eval.tracking import REPORTS_DIR, git_commit

SCHEMA = "loadtest"
PLACES = [("Chicago", "Illinois", "United States"), ("Tokyo", "Tokyo", "Japan"), ("Paris", "Île-de-France", "France"),
          ("Columbus", "Ohio", "United States"), ("Lisbon", "Lisbon", "Portugal")]  # fmt: skip
CAMERAS = [("Canon EOS R6", "RF50mm F1.8 STM", 50.0), ("Fujifilm X100V", None, 23.0),
           ("Apple iPhone 15 Pro", "iPhone 15 Pro back camera", 6.8), ("Sony ILCE-7M3", "FE 85mm F1.8", 85.0)]  # fmt: skip
QUERIES = ["dog on a beach", "city at night", "mountain lake", "birthday cake", "street photography",
           "sunset over the ocean", "snowy forest", "people dancing", "coffee cup", "red car"]  # fmt: skip


def run(n: int, dim: int, batch: int = 5000, seed: int = 0) -> dict[str, Any]:
    from psycopg import sql

    from api.db.migrate import MIGRATIONS_DIR
    from api.db.session import close_pool, connect
    from api.db.vector_index import ensure_image_index

    s = get_settings()
    rng = np.random.default_rng(seed)
    base_url = s.database_url
    with connect(base_url, vector=False) as conn:
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(SCHEMA)))
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(SCHEMA)))
        conn.execute(sql.SQL("SET search_path = {}, public").format(sql.Identifier(SCHEMA)))
        for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
            conn.execute(f.read_text().replace("CREATE EXTENSION IF NOT EXISTS vector;", ""))
        conn.commit()

    url = base_url + ("&" if "?" in base_url else "?") + f"options=-csearch_path%3D{SCHEMA},public"
    model = "loadtest-clip"
    t0 = time.time()
    start = datetime(2015, 1, 1, tzinfo=UTC)
    with connect(url) as conn:
        for off in range(0, n, batch):
            m = min(batch, n - off)
            ids = [uuid.uuid4() for _ in range(m)]
            vecs = rng.standard_normal((m, dim)).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            with conn.cursor() as cur:
                with cur.copy(
                    "COPY photos (id, path, file_hash, taken_at, place_name, admin1, country, camera, lens, "
                    "focal_length, width, height, thumb_path) FROM STDIN"
                ) as cp:
                    for i, pid in enumerate(ids):
                        pl = PLACES[rng.integers(len(PLACES))]
                        cam = CAMERAS[rng.integers(len(CAMERAS))]
                        cp.write_row((pid, f"/synthetic/{off + i}.jpg", f"h{off + i}",
                                      start + timedelta(minutes=int(rng.integers(0, 10 * 365 * 24 * 60))),
                                      *pl, *cam, 4000, 3000, "/dev/null"))  # fmt: skip
                with cur.copy(
                    "COPY image_embeddings (photo_id, model, embedding) FROM STDIN WITH (FORMAT BINARY)"
                ) as cp:
                    cp.set_types(["uuid", "text", "vector"])
                    for pid, v in zip(ids, vecs, strict=True):
                        cp.write_row((pid, model, v))
            conn.commit()
        load_s = time.time() - t0
        t1 = time.time()
        conn.execute("SET maintenance_work_mem = '1GB'")
        ensure_image_index(conn, model, dim)
        index_s = time.time() - t1
        conn.execute("ANALYZE")
        conn.commit()
        sizes = conn.execute(
            "SELECT pg_total_relation_size('photos') AS photos, pg_total_relation_size('image_embeddings') AS emb, "
            "pg_relation_size(%s::regclass) AS hnsw",
            (f"{SCHEMA}.image_emb_hnsw_loadtest_clip",),
        ).fetchone()

    # latency through the real query path: CLIP text encoder + the engine's SQL
    import os

    os.environ["PS_DATABASE_URL"] = url
    get_settings.cache_clear()
    close_pool()
    from api.ml.clip import get_clip
    from api.search import engine
    from api.search.filters import SearchFilters

    enc = get_clip()
    engine.reset_caches()
    results = {}
    for label, flt in (("no filters", None), ("place filter", SearchFilters(place="Tokyo")),
                       ("place + year", SearchFilters(place="Tokyo", date_from="2020-01-01", date_to="2020-12-31"))):  # fmt: skip
        lat, enc_lat = [], []
        for rep in range(3):
            for q in QUERIES:
                te = time.perf_counter()
                qv = enc.encode_texts([q])[0]
                enc_ms = (time.perf_counter() - te) * 1000
                from api.db.session import get_conn

                with get_conn() as conn:
                    where, params = engine.flt.to_sql(flt)
                    exact = (
                        bool(flt)
                        and engine._count_filtered(conn, where, params) <= engine.EXACT_SCAN_MAX
                    )
                    engine._ann_settings(conn, exact, 200)
                    ts = time.perf_counter()
                    engine.clip_candidates(conn, qv, model, dim, where, params, 200, exact)
                    sql_ms = (time.perf_counter() - ts) * 1000
                if rep:  # first pass warms caches
                    lat.append(enc_ms + sql_ms)
                    enc_lat.append(enc_ms)
        results[label] = {
            "p50_ms": percentile(lat, 50),
            "p95_ms": percentile(lat, 95),
            "encode_p50_ms": percentile(enc_lat, 50),
        }
    os.environ["PS_DATABASE_URL"] = base_url
    get_settings.cache_clear()
    close_pool()
    return {
        "kind": "load_test",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "n": n,
        "dim": dim,
        "clip_model_for_queries": enc.spec.name,
        "insert_s": round(load_s, 1),
        "hnsw_build_s": round(index_s, 1),
        "storage_mb": {k: round(v / 1e6, 1) for k, v in dict(sizes or {}).items()},
        "latency": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--keep", action="store_true", help="keep the loadtest schema afterwards")
    a = ap.parse_args()
    out = run(a.n, a.dim)
    if not a.keep:
        from api.db.session import connect

        with connect(get_settings().database_url, vector=False) as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
            conn.commit()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_loadtest_{a.n}.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
