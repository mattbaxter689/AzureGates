FROM python:3.11-slim

# Pin to a specific uv version for reproducibility
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    # This is the key: installs into system Python, not a .venv
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Non-root user
RUN useradd --create-home --shell /bin/bash mluser && \
    chown -R mluser:mluser /app
USER mluser

ENTRYPOINT ["python"]
CMD ["--help"]
