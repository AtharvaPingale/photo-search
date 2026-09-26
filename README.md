# Photo Search

Search your own photo library in plain language ("golden hour on the beach", "street shots at night with the 35mm", "me and Rohan in Chicago") from your computer or your phone. Everything runs on one home machine: no photos leave it.

It started as CLIP semantic search and grew into a hybrid system: a query parser that turns "last summer with the 50mm" into SQL filters, VLM captions and OCR fused with reciprocal rank fusion, face clustering, near-duplicate and burst detection, auto albums, an agent that answers questions about the library with photo evidence, a CLIP fine-tuned on your own photos, and a PWA you can install on your phone. Every feature is measured against a hand-labelled eval set (see [Evaluation](#evaluation)).

```
 photo folders ──watchdog──▶ ingest queue (Redis) ──▶ EXIF/GPS · CLIP · captions+OCR · faces
 phone backup ──auto import─┘                                    │
                                                        PostgreSQL + pgvector
                                                                 │
                                     FastAPI: search / agent / labelling / media
                                                                 │
                 React PWA ◀── home network / Tailscale ──▶ your phone
```

## Contents

- [Quick start (Docker)](#quick-start-docker)
- [Native setup](#native-setup-no-docker)
- [Using it from your phone](#using-it-from-your-phone)
- [How search works](#how-search-works)
- [Evaluation](#evaluation) and [results](#results)
- [Faces and privacy](#faces-and-privacy)
- [Organisation: duplicates, bursts, albums](#organisation)
- [The agent](#the-agent)
- [Fine-tuning CLIP on your library](#fine-tuning-clip-on-your-library)
- [Auto import, backups, moving to a new machine](#auto-import-backups-and-moving-to-a-new-machine)
- [Scale](#scale)
- [Configuration](#configuration) · [Development](#development) · [Troubleshooting](#troubleshooting)

## Quick start (Docker)

Needs Docker with GPU support: on Windows, Docker Desktop with WSL integration enabled (Settings → Resources → WSL Integration); on Linux, the NVIDIA container toolkit. The `worker` service reserves the GPU; on a machine without one, delete its `deploy:` block and everything runs on CPU, slowly. The query parser and agent also want [Ollama](https://ollama.com) on the host with a model that supports tools.

```bash
ollama pull qwen2.5:14b-instruct
cp .env.example .env          # set PHOTOS_DIR; optionally IMPORT_DIR, PS_AUTH_TOKEN, PS_PORT, DATA_DIR
make up                       # builds the image, starts db, redis, api, gpu worker, ingest worker, watcher, nightly backups
docker compose exec api photo-search index --queue   # first full index; the watcher handles new photos after that
```

Open http://localhost:8000 (or `PS_PORT`). Progress: `docker compose logs -f worker`, `docker compose exec api photo-search db stats`.

Notes from running it on Windows 11 + WSL2 + Docker Desktop:

- The first build takes 10-15 minutes; Docker Desktop reports the image at ~15 GB (about 10 GB of layers), most of it the CUDA libraries GPU torch needs.
- On Windows, keep the repo inside WSL and point `PHOTOS_DIR` at the library wherever it lives, e.g. `/mnt/c/Users/you/Pictures`. The watcher polls rather than relying on file events, which Windows drives don't deliver to Linux; a new photo was searchable 36 s after it landed.
- The containers run as root, so files they create under `DATA_DIR` and `BACKUPS_DIR` (default `./data`, `./backups`) are root-owned on the host. Point both at a dedicated folder outside the repo if that bothers you.
- The API runs the text encoder on CPU on purpose (the GPU belongs to the worker); warm searches still take tens of milliseconds.

The first index of a large library is dominated by captioning; every stage is resumable, so stop and start it whenever you like. Measured on 300 photos with an RTX 5070 Ti (times include loading each model): scan + thumbnails + CLIP + Florence-2 captions 53 s, OCR 31 s (the text gate sent 23 of the 300 photos to OCR), face detection 32 s.

## Native setup (no Docker)

```bash
uv sync --extra all            # Python 3.12; CUDA 12.8 torch wheels
cd web && npm ci && npm run build && cd ..
# Postgres 16+ with pgvector, e.g. `docker run -d -p 5432:5432 -e POSTGRES_USER=photos -e POSTGRES_PASSWORD=photos pgvector/pgvector:pg17`
export PS_DATABASE_URL=postgresql://photos:photos@localhost:5432/photos
export PS_PHOTO_ROOTS='["/path/to/Pictures"]'
uv run photo-search db migrate
uv run photo-search index      # scan -> video frames -> CLIP -> captions -> OCR, inline (no Redis needed)
uv run photo-search serve      # http://127.0.0.1:8000
uv run photo-search watch --inline   # keep new photos searchable within a minute
```

Everything is a subcommand of `photo-search` (`--help` lists them) and most have a `make` target (`make help`).

## Using it from your phone

The web app is a PWA: open it in Safari/Chrome on the phone and "Add to Home Screen". It has an infinite-scroll grid of 256 px WebP thumbnails (~8 KB each, cached forever by the service worker since their URLs carry the content hash), a swipeable lightbox that loads a 2048 px rendition first and the full-resolution original on demand, voice search (the browser's speech API; the audio goes to Apple's/Google's recogniser, photos never do), and Share / Save buttons that hand the photo itself to the OS share sheet.

Nothing is exposed publicly. Two ways to reach it:

1. **Tailscale (works on mobile data).** Install Tailscale on the server and the phone, then on the server:
   ```bash
   tailscale serve --bg 8000      # https://<machine>.<tailnet>.ts.net -> localhost:8000, with a real TLS cert
   ```
   Or without installing Tailscale on the host: set `TS_AUTHKEY` in `.env` and `docker compose --profile tailscale up -d`.
2. **Home Wi-Fi only.** Set `PS_BIND=0.0.0.0` and open `http://<server-ip>:8000`.

Either way, set a token so only your devices get in:

```bash
uv run photo-search token                  # put the output in .env as PS_AUTH_TOKEN, restart the api
uv run photo-search pair https://photos.your-tailnet.ts.net   # prints a QR code; scan it with the phone camera
```

The pairing link carries the token in the URL fragment (never sent to the server or logged); the app exchanges it once for an HttpOnly cookie. Scripts can use `Authorization: Bearer <token>`. Rotating the token logs every device out.

## How search works

```
"sunset in Tokyo last summer with the X100V"
   │ query parser (rules + LLM)                      filters: place=Tokyo, dates=2026-06-01..08-31, camera=X100V
   ▼                                                 semantic: "sunset"
 CLIP text vector ─┐
 caption vector ───┼─ each ranks candidates *inside* the SQL filters ─▶ weighted RRF ─▶ results
 keyword tsquery ──┤                                                    (videos collapse to their best frame)
 OCR tsquery ──────┘
```

- **Query parser** (`api/search/query_parser.py`). A deterministic rule parser driven by your library's own vocabulary (the cameras, lenses, places and people that actually exist in the database) handles dates, seasons, holidays, focal lengths, lens names, apertures, ISO, people and places in ~0.1 ms. Queries with metadata cues also go to an LLM with a strict JSON schema; its output is then *validated*: every filter needs evidence in the query text and a match in the library, and date arithmetic is taken from the rules. See the parser results for why.
- **Filters first.** Filters are SQL `WHERE` clauses with bound parameters. When they select fewer than 20k photos the ranking is an exact scan (a few ms, and it never loses results to HNSW post-filtering); otherwise the per-model HNSW index is used.
- **Fallback.** If filters match nothing, the search reruns as pure semantic search on the full query and the UI says so.
- **Editable chips.** What the parser extracted shows up as chips; tap to edit or remove, and the search reruns with your filters.
- **Search by example / relevance feedback.** "Similar" searches by a photo's embedding; upload a photo to search with it. 👍/👎 in the lightbox apply Rocchio feedback to the query vector and are logged (they later become fine-tuning pairs).
- **Latency.** Warm CLIP text encoding is ~5 ms on GPU (~20 ms CPU); retrieval SQL is a few ms at 50k photos. The LLM parser adds ~2 s with a local 14B model, which is why it only runs when the rules leave metadata cues unexplained (7 of 62 parser-eval queries); measured end to end on the demo library, typical searches take 20-40 ms warm.

## Evaluation

Every change is judged by its effect on a labelled query set, so it came before any of the fancy features.

- **Labelling tool**: the *Eval* tab (Library → Eval labelling). Pick or write a query, tap photos to mark them relevant (tap again for "highly relevant", used by nDCG), save. Candidates are pooled from several systems (CLIP alone, captions, keywords/OCR, the fused ranking, plus any alternative phrasings you try), so labels aren't biased toward whatever the current best system already finds.
- **Query set**: `eval/queries/queries.jsonl` (in git). It starts with 150 candidate queries across six categories (objects, attributes, metadata, text, people, hard). Edit them to fit your library, label them, add a query whenever a real search disappoints you. Photos are referenced by content hash, so labels survive re-indexing and moved folders.
- **Splits**: each query is assigned to *dev* (70%) or *test* (30%) from a hash of its id, once. Tuning code only ever reads dev; `make eval-test` is for final numbers.
- **Metrics**: Recall@5, Recall@20, MRR, nDCG@10 per category, 95% bootstrap confidence intervals, p50/p95 latency (end to end, retrieval only, parser).

```bash
make eval NAME=baseline          # dev split; report in eval/reports/, diffed against the previous run, logged to MLflow
make tune-fusion                 # grid-search RRF weights on dev (5-fold CV score reported), writes eval/fusion_weights.json
make ablation                    # CLIP only / + parser / + captions / + OCR / + tuned fusion, per category
make compare-models MODELS=openclip-vitb32,openclip-vitl14,finetuned-v1
make parser-eval agent-eval face-eval dup-eval load-test
make mlflow                      # http://localhost:5000
```

Each report records the git commit, config and MLflow run id; the markdown lists the ten worst queries so failures get looked at, not just averages.

## Results

The retrieval, agent, face and duplicate numbers depend on your library and your labels, so they get filled in from your own runs: paste the markdown from `eval/reports/ablation_dev.md` (and `_test.md` at the end) here, each row carries its MLflow run id.

### Query parser

These don't depend on the library: 62 labelled queries with their own "today" and a fixed vocabulary (`eval/queries/parser_labels.jsonl`), local `qwen2.5:14b-instruct` via Ollama.

Field accuracy / exact match / p50 latency. MLflow run ids in brackets (experiment `query-parser`).

| parser | tuning set (62 queries) | held-out set (25 queries, never tuned on) |
|---|---|---|
| `rules` | 0.976 / 0.968 / 0.1 ms `[7255b4cd]` | 0.946 / 0.960 / 0.1 ms `[1005cd5f]` |
| `llm_raw` (schema-constrained, unvalidated) | 0.364 / 0.081 / 1.8 s `[d3ebb8cc]` | 0.458 / 0.080 / 2.1 s `[125137b3]` |
| `llm` (validated) | 0.720 / 0.774 / 2.0 s `[23573662]` | 0.829 / 0.800 / 2.0 s `[f0c63555]` |
| `hybrid` (rules + validated LLM, always) | 0.988 / 0.984 / 1.9 s `[f599bdfe]` | 0.921 / 0.920 / 2.0 s `[d08a933b]` |
| `auto` (default: LLM only for what rules leave unexplained) | 0.976 / 0.968 / 0.1 ms (p95 1.9 s) `[961f60d6]` | 0.946 / 0.960 / 0.1 ms (p95 1.6 s) `[1956aaa4]` |

Held-out failures kept as they are (fixing them now would turn the held-out set into a tuning set): rules read "christmas lights 2023" as all of 2023 instead of Christmas 2023; the LLM guessed a lake in "the drone over the lake" was South Lake Tahoe and turned "Ohio in the fall" into this autumn only.

What this showed, and what changed because of it:

- **The raw LLM is worse than regexes.** Asked for a strict JSON schema, the 14B model still added `media: "photo"` to nearly every query, turned "beach" into a place, answered "Tokyo, Japan" instead of "Tokyo", invented a camera from a lens mention, and got relative dates wrong ("last summer" in September → the previous year). Schema-constrained decoding guarantees the *shape*, not the *content*.
- **Validation fixed most of it**: every LLM filter must be evidenced in the query and exist in the library, so the model proposes and code disposes.
- **Rules beat the validated LLM**, and the always-on hybrid is *worse* than rules on the held-out set (0.921 vs 0.946): the LLM adds more wrong guesses than it fixes. The rules were written while looking at the tuning set, and they drop from 0.976 to 0.946 on queries written afterwards, the size of that optimism.
- **Latency decides the architecture**: rules take ~0.1 ms, the LLM ~1.8 s. The default `auto` mode always runs the rules and calls the LLM only when cues remain in what the rules left over ("cats in Chicago" never reaches it), merging with rules winning on conflicts. It matches rules on accuracy at a 0.1 ms median.

### Demo smoke test

Before your library is labelled, a sanity check on 300 COCO val2017 photos (synthetic EXIF, query = a human-written COCO caption, relevant = that one photo) shows the pipeline behaves as expected end to end with the real models. It is not a substitute for your eval: one relevant photo per query and web photos, not a personal library.

nDCG@10 on the dev split (109 queries; 51 more held out), local Florence-2 captions, PaddleOCR, `qwen2.5:14b-instruct` for the parser:

| configuration | all | metadata | objects | p50 ms |
|---|---:|---:|---:|---:|
| CLIP only | 0.881 | 0.830 | 0.899 | 7 |
| CLIP + query parser | 0.908 | 0.947 | 0.894 | 8 |
| + captions (embedding + keywords), equal weights | 0.862 | 0.947 | 0.832 | 15 |
| + OCR, equal weights | 0.858 | 0.947 | 0.827 | 16 |
| + tuned fusion (clip 1, caption 0.5, ocr 0.25, keyword 0; RRF k=20) | 0.907 | 0.960 | 0.889 | 15 |

The parser does what it's for (metadata 0.830 → 0.947). Captions *hurt* at equal weight here and tuned fusion only ties CLIP + parser: COCO's queries are exactly the kind of caption CLIP was trained on, and keyword search over Florence's long captions is pure noise (its tuned weight is 0). Tuning chose these weights by 5-fold CV on dev (0.903 vs 0.867 for equal weights). Whether captions earn their GPU hours on a personal library, with its counts, actions and text queries, is what your own eval has to show.

Also found by this run and fixed: the default fusion weights referenced a signal name that doesn't exist; the tuner's "CLIP only" baseline silently gave other signals weight 1; the geocoder tagged photos with neighbourhoods ("Chicago Loop", "Hatsudai") so "in Tokyo" found nothing; the agent date-sorted before cutting its result list and answered "no giraffe photos" when there were ten.

## Faces and privacy

- Opt-in per folder: faces are only detected under `PS_FACE_ROOTS`. Unset = never.
- InsightFace (`buffalo_l`: SCRFD detection + ArcFace embeddings) on the GPU, HDBSCAN clustering. Re-clustering keeps names (new clusters inherit the id of the old cluster they overlap most) and never overrides manual edits.
- People tab: name a cluster (and aliases such as "me"), select several to merge, select faces to split them off or mark "not this person", assign unclustered faces.
- Then "me and Rohan hiking" filters to photos containing both.
- `photo-search faces wipe` deletes every detection, embedding, cluster, name and crop.
- Quality: `python -m eval.face_eval sample` writes a CSV of faces to label by name; `make face-eval` reports purity, completeness, coverage (share of faces that got a cluster) and pairwise precision/recall.

## Organisation

`photo-search organize` (or `make organize`), then the Organize and Albums pages:

- **Near-duplicates** need both signals to agree: CLIP cosine ≥ 0.95 and perceptual-hash distance ≤ 10 bits. The suggested keeper is the largest, then sharpest. `make dup-eval` measures precision/recall on labelled pairs; the sampler mixes flagged pairs with near misses so recall is measurable too.
- **Bursts**: same camera, ≤ 2 s apart, similar. Best pick = highest variance of the Laplacian (sharpness).
- **Auto albums**: the timeline is split at 8-hour gaps, "home" is the region with the most photo-days, consecutive away-from-home segments merge into trips; the LLM writes a title and summary from places, dates and captions (template fallback).

## The agent

The Ask tab (or `photo-search ask "..."`): a LangGraph agent with four tools. It never writes SQL:

| tool | what it does |
|---|---|
| `semantic_search(query, filters, sort)` | the normal search engine; `sort="newest"` answers "when did I last…" |
| `metadata_query(template, params)` | one of five named, parameterised SQL templates (photos in range, first & last, on a date, albums in range, library overview) |
| `aggregate(group_by, filters)` | counts by city / country / camera / lens / year / month / weekday / hour / focal length / person (whitelisted SQL expressions) |
| `get_photo_details(id)` | metadata, caption, OCR text, people |

Arguments are pydantic-validated before touching SQL; values are bound parameters. The final step asks for `{answer, evidence: [{photo_id, note}]}` and drops any cited id that no tool actually returned; an answer is *grounded* only if it cites real photos and invented none. LangSmith traces the graph, each LLM call and each tool call when `LANGSMITH_TRACING=true`.

`python -m eval.agent_eval make_qa` generates short-answer questions whose ground truth comes from your library via the same safe templates (not via the agent), e.g. "Which city did I photograph most in 2025?"; add free-form ones by hand. `make agent-eval` scores correctness (string match + LLM judge), groundedness, hallucinated citations and tool calls per answer; `python -m eval.agent_eval rate` collects your own grades so the report can show judge-vs-human agreement and Cohen's kappa.

## Fine-tuning CLIP on your library

```bash
make pairs        # training/build_pairs.py
make finetune     # LoRA fine-tune -> data/models/finetuned-v1, then embed the library with it
make compare-models MODELS=openclip-vitb32,finetuned-v1   # before/after per category, dev split
```

Training pairs come from VLM captions (first sentences, boilerplate stripped), Lightroom/XMP keywords ("a photo of beach, sunset, Portugal") and 👍 feedback clicks. Two modes: `--mode lora` (rank-8 adapters on every attention and MLP weight via torch parametrizations, which also works on `nn.MultiheadAttention`'s fused `in_proj_weight`; merged into the weights on save) and `--mode last-layers --unfreeze N`. `--wise-alpha` interpolates back toward the base weights (WiSE-FT) to keep CLIP's general knowledge; choose alpha on dev. Embeddings are stored under the new model name next to the base ones, so nothing is re-indexed to compare.

**Leakage precautions** (counted in `data/training/manifest.json`):

1. Every photo labelled relevant for *any* eval query (dev or test) is excluded from training, along with its near-duplicate and burst group and anything shot within 10 minutes on the same camera. The same scene photographed twice is not an independent example.
2. Feedback queries that match an eval query (normalised text, or token Jaccard ≥ 0.6) are dropped; training on the eval query text itself would inflate exactly the number being measured.
3. The in-training validation split is by day, not by photo, so near-identical shots never straddle it.
4. Fusion weights and WiSE alpha are chosen on dev; test is run once at the end.

Report the per-category table honestly, including categories that got worse (fine-tuning on captions tends to help objects/scenes and can hurt text-in-image and hard queries).

## Auto import, backups and moving to a new machine

**Auto import.** Point `IMPORT_DIR` (Docker) or `PS_IMPORT_DIRS` at your phone's backup folder (PhotoSync, Syncthing, a synced iCloud/Google Photos folder). New files are copied (or moved, `PS_IMPORT_MODE=move`) into `Imported/YYYY/YYYY-MM/` and indexed. Duplicates are detected by content hash (backup apps love re-uploading "IMG_0001 (1).HEIC"); a ledger remembers every file ever imported, so deleting a photo from the library doesn't make it reappear on the next sync; files still being uploaded are left for the next sweep; XMP/AAE sidecars come along. The watcher sweeps on every change and every 10 minutes; `photo-search import` runs it once.

**Backups.** The `backup` service runs `pg_dump` nightly at `BACKUP_HOUR` into `./backups`, keeping `PS_BACKUP_KEEP_DAYS` (14) days and always the newest. `make backup` takes one now. The database is what matters (embeddings and captions are hours of GPU time; names, merges and albums are your work). Thumbnails are rebuilt by `photo-search repair`, eval labels are in git, and your photo library itself needs its own backup.

**Restore**: `make restore` (newest) or `make restore BACKUP=backups/photos-20260101-030000.dump`. It stops the writers, restores, re-applies migrations, restarts and rebuilds missing thumbnails. Natively: `photo-search db restore [file]`.

**Rebuild on a new machine:**

```bash
git clone <this repo> && cd photo-search
cp /old/.env .env && cp -r /old/backups .   # or copy just the newest .dump
make up && make restore
# if the library lives somewhere else now:
docker compose exec api photo-search relink /old/library/path /photos
docker compose exec api photo-search repair       # thumbnails
```

Thumbnails, frames and face crops are stored relative to the data directory, so only the originals' paths ever need `relink`.

## Scale

`make load-test` inserts 100,000 synthetic photos (random unit-norm 512-d embeddings, plausible metadata) into a separate schema and measures insert time, HNSW build time, storage and query latency through the real code path (CLIP text encoder + the engine's SQL, with and without filters).

100,000 photos, 512-d, embedded Postgres 16 + pgvector on the dev machine:

| | |
|---|---|
| bulk insert (COPY) | 2.6 s |
| HNSW build (m=16, ef_construction=128) | 40 s |
| storage | photos 50 MB, embeddings 566 MB (incl. the 273 MB HNSW index) |
| query p50 / p95, no filters (HNSW) | 17 / 19 ms |
| query p50 / p95, place filter (~20k matches, exact scan) | 90 / 110 ms |
| query p50 / p95, place + year | 19 / 21 ms |

CLIP text encoding is ~4 ms of each. The slowest case is a filter that keeps ~20% of the library: small enough to be ranked exactly, large enough to cost a sequential distance computation; still well under the 300 ms target.

## Configuration

All settings are environment variables with the `PS_` prefix (or `.env`); `.env.example` documents them. The important ones: `PS_PHOTO_ROOTS`, `PS_FACE_ROOTS`, `PS_IMPORT_DIRS`, `PS_AUTH_TOKEN`, `PS_IMAGE_MODEL` (`openclip-vitb32`, `openclip-vitl14` or a fine-tuned name), `PS_CAPTION_BACKEND` (`florence2`, `qwen2vl`, `anthropic` = opt-in cloud captions that send thumbnails to the API, `none`), `PS_LLM_PROVIDER` (`ollama` = local, `anthropic` = Claude via the official SDK, text only, `none` = rules only).

## Development

```bash
make test    # 77 tests against a real embedded Postgres + pgvector (pgserver), fake deterministic models; no GPU, network or Docker
make lint    # ruff, mypy, tsc
cd web && npm run dev   # UI on :5173 with /api proxied to :8000
```

```
api/        FastAPI app, config, auth, LLM clients
  search/   query parser, filters -> SQL, retrieval + fusion engine
  agent/    LangGraph graph and the four tools
  db/       migrations, connection pool, backups
  routes/   HTTP endpoints
  ml/       CLIP and text-embedding wrappers, model registry
workers/    ingest (scan/EXIF/geocode/thumbnails), embed, caption, ocr, faces, video, organize,
            importer, watch, queue (RQ), maintenance (relink/repair)
training/   build_pairs.py, finetune_clip.py
eval/       queries/ (labelled sets), run_eval, tune_fusion, ablation, parser/agent/face/dup evals, load_test, reports/
web/        React + TypeScript PWA
tests/
```

## Troubleshooting

- **WSL: `docker` not found.** Enable WSL integration in Docker Desktop (Settings → Resources → WSL Integration).
- **Port 8000 already in use.** Set `PS_PORT=8001` (or any free port) in `.env`.
- **New photos on a Windows drive (`/mnt/c`) aren't picked up.** WSL doesn't deliver file events for Windows drives; use `photo-search watch --poll` (the Docker watcher already polls).
- **The LLM parser or agent says "ollama not reachable".** Start Ollama on the host; from Docker it's reached at `host.docker.internal:11434`. `PS_LLM_PROVIDER=none` runs with rules only.
- **GPU not used.** `uv run python -c "import torch; print(torch.cuda.is_available())"`; in Docker the `worker` service reserves the GPU and needs the NVIDIA container toolkit. The API runs the text encoder on CPU on purpose.
- **OpenCV import errors.** Only `opencv-contrib-python` may be installed (PaddleX checks for it by name); `pyproject.toml` excludes the other variants insightface and friends pull in. If `cv2` breaks after an install, `uv sync --reinstall-package opencv-contrib-python`.
