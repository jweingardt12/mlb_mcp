# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy source and install the project itself
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
FROM python:3.12-alpine AS runtime

WORKDIR /app
COPY --from=builder /app /app

# Smithery sets PORT=8081 for hosted servers
ENV PATH="/app/.venv/bin:$PATH" \
    TRANSPORT=http \
    HOST=0.0.0.0 \
    PORT=8081

EXPOSE ${PORT}

# Health-check hits the /health endpoint added in server.py
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:${PORT}/health || exit 1

CMD ["python", "-m", "mlb_mcp.server"]
