# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/app/models/huggingface \
    TORCH_HOME=/app/models/torch \
    XDG_CACHE_HOME=/app/models/cache

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libsndfile1 \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

RUN mkdir -p /app/models /app/data /app/logs /app/tmp \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app \
    && chmod 1777 /app/tmp \
    && chmod -R a+rX /opt/venv /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3.12 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').getenv('API_PORT', '8000') + '/health/live', timeout=3)"

CMD ["sh", "-c", "/opt/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${API_PORT:-8000} --workers 1"]
