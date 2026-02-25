FROM nvidia/cuda:12.2.0-base-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

ENV UV_HTTP_TIMEOUT=1000 \
    UV_HTTP_RETRIES=10 \
    UV_SYSTEM_PYTHON=true \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      python3 python3-venv python3-dev \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock .python-version ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY *.py /app/

ENV TRANSCRIBATION_API_HOST=0.0.0.0 \
    TRANSCRIBATION_API_PORT=11430

EXPOSE 11430

ENTRYPOINT uv run uvicorn main:app --host ${TRANSCRIBATION_API_HOST} --port ${TRANSCRIBATION_API_PORT}
