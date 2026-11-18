FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Dependencies first: this layer is cached as long as the lock file is unchanged.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY eval ./eval

RUN useradd --create-home --uid 1000 bot && mkdir -p /models && chown bot:bot /models
USER bot

CMD ["python", "-m", "app.bot"]
