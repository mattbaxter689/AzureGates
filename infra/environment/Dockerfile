FROM python:3.11-slim AS builder

# Install uv — the Rust-based Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# uv respects this to avoid creating a venv inside the venv
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy dependency manifest first for layer-cache efficiency
COPY pyproject.toml uv.lock ./

# Install all dependencies into /usr/local (system Python)
# --no-dev skips test extras; adjust if you want them in the image
RUN uv pip install --system .

# Non-root user for security
RUN useradd --create-home --shell /bin/bash mluser && \
    chown -R mluser:mluser /app
USER mluser

# Default entrypoint (overridden per gate by AML Command Jobs)
ENTRYPOINT ["python"]
CMD ["--help"]
