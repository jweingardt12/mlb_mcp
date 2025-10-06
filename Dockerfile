FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Ensure consistent locales and behavior
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Install system build essentials for wheels that might require compilation
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv from PyPI to avoid pulling the ghcr.io/astral-sh/uv image
RUN pip install --upgrade pip && pip install --no-cache-dir "uv==0.5.16"

# Copy project metadata first for dependency resolution caching
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY smithery.yaml ./

# Install project dependencies (and the project itself) into a virtual environment
RUN if [ -f "uv.lock" ]; then \
        uv sync --frozen --no-dev; \
    else \
        uv sync --no-dev; \
    fi

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

# Copy the entire application along with the prepared virtual environment
COPY --from=builder /app /app

# Default command: start the Smithery MCP server
CMD ["smithery", "start"]
