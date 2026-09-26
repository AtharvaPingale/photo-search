.DEFAULT_GOAL := help
RUN := uv run
NAME ?= run
SPLIT ?= dev

help:  ## list targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup
install:  ## Python deps (all extras) + web deps + git hooks
	uv sync --extra all
	cd web && npm ci
	$(RUN) pre-commit install

up:  ## start everything in Docker (db, redis, api, workers, watcher, backups)
	docker compose up -d --build

down:  ## stop Docker services
	docker compose down

logs:  ## follow Docker logs
	docker compose logs -f --tail=100

migrate:  ## apply database migrations
	$(RUN) photo-search db migrate

# ---------------------------------------------------------------- indexing
index:  ## scan + embed + caption + OCR inline (no Redis needed)
	$(RUN) photo-search index

index-queue:  ## same, via the Redis job queue
	$(RUN) photo-search index --queue

faces:  ## detect + cluster faces (only under PS_FACE_ROOTS)
	$(RUN) photo-search faces detect && $(RUN) photo-search faces cluster

organize:  ## duplicates, bursts, auto albums
	$(RUN) photo-search organize

watch:  ## index new photos as they appear (+ auto import)
	$(RUN) photo-search watch --inline

# ---------------------------------------------------------------- run
serve:  ## API + built web app on :8000
	$(RUN) photo-search serve

web-dev:  ## Vite dev server on :5173 (proxies /api to :8000)
	cd web && npm run dev

web-build:  ## build the PWA into web/dist (served by the API)
	cd web && npm run build

pair:  ## QR code to log a phone in: make pair URL=https://photos.tailnet.ts.net
	$(RUN) photo-search pair $(URL)

# ---------------------------------------------------------------- quality
test:  ## Python tests (embedded Postgres, fake models)
	$(RUN) pytest -m "not slow"

lint:  ## ruff + mypy + TypeScript
	$(RUN) ruff check api workers eval training tests
	$(RUN) ruff format --check api workers eval training tests
	$(RUN) mypy api workers eval training
	cd web && npx tsc --noEmit

# ---------------------------------------------------------------- evaluation
eval:  ## retrieval eval on the dev split, compared with the previous run: make eval NAME=captions
	$(RUN) python -m eval.run_eval --name $(NAME) --split $(SPLIT)

eval-test:  ## held-out test split: final numbers only, never tune on these
	$(RUN) python -m eval.run_eval --name $(NAME) --split test

tune-fusion:  ## tune RRF weights on dev (writes eval/fusion_weights.json)
	$(RUN) python -m eval.tune_fusion

ablation:  ## CLIP only / + parser / + captions / + OCR / + tuned fusion
	$(RUN) python -m eval.ablation --split $(SPLIT)

compare-models:  ## per-category table across image models: make compare-models MODELS=openclip-vitb32,finetuned-v1
	$(RUN) python -m eval.ablation --split $(SPLIT) --models $(MODELS)

parser-eval:  ## query parser field accuracy: rules vs LLM vs hybrid
	$(RUN) python -m eval.parser_eval

agent-eval:  ## agent accuracy, groundedness, tool calls
	$(RUN) python -m eval.agent_eval run

face-eval:  ## face clustering purity / coverage from eval/queries/face_labels.csv
	$(RUN) python -m eval.face_eval score

dup-eval:  ## duplicate detection precision / recall from eval/queries/dup_pairs.csv
	$(RUN) python -m eval.dup_eval score

load-test:  ## 100k synthetic photos: index build time, latency, storage
	$(RUN) python -m eval.load_test --n 100000

mlflow:  ## MLflow UI on :5000
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ---------------------------------------------------------------- fine-tuning
pairs:  ## training pairs from captions, keywords, feedback (eval photos excluded)
	$(RUN) python -m training.build_pairs

finetune:  ## LoRA fine-tune of CLIP, saved as finetuned-v1
	$(RUN) python -m training.finetune_clip --mode lora --name finetuned-v1
	$(RUN) photo-search embed --model finetuned-v1

# ---------------------------------------------------------------- backups
backup:  ## back up the database now (Docker)
	docker compose exec -T backup sh -c 'pg_dump --format=custom --compress=6 --file=/backups/photos-$$(date +%Y%m%d-%H%M%S).dump'
	docker compose exec -T backup sh -c 'ls -1t /backups/*.dump | head -n 1'

restore:  ## restore the newest backup, or: make restore BACKUP=backups/photos-....dump
	sh scripts/restore.sh $(BACKUP)

.PHONY: help install up down logs migrate index index-queue faces organize watch serve web-dev web-build pair test lint \
	eval eval-test tune-fusion ablation compare-models parser-eval agent-eval face-eval dup-eval load-test mlflow \
	pairs finetune backup restore
