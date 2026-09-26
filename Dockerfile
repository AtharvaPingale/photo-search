# One image for the API and all workers. Build: `docker compose build`.

FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# libgomp: PaddleOCR/torch CPU kernels; libgl1 + libglib: OpenCV (the contrib build PaddleX requires).
# ffmpeg, libheif and libraw ship inside wheels.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# the uv download cache lives in a BuildKit cache mount, not in the image (~10 GB)
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --extra all --no-dev --no-install-project
COPY api ./api
COPY workers ./workers
COPY training ./training
COPY eval ./eval
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --extra all --no-dev
COPY --from=web /web/dist ./web/dist
ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8000
CMD ["photo-search", "serve", "--host", "0.0.0.0", "--port", "8000"]
