# Multi-stage Dockerfile for Smithery-compatible MCP server
FROM python:3.11-slim-bookworm as builder

# Install build tools and libraries needed for packages like pandas and lxml
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libxml2-dev \
        libxslt1-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim-bookworm

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libxml2 \
        libxslt1.1 \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --shell /bin/bash app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/app/.local

# Make sure scripts in .local are usable
ENV PATH=/home/app/.local/bin:$PATH

WORKDIR /app

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Change ownership to app user
RUN chown -R app:app /app

USER app

# EXPOSE is informational; Smithery uses the $PORT env var that Uvicorn will listen on.
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/ || exit 1

# Set Python optimization flags for better performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PYTHONIOENCODING=utf-8

# Use the startup script
CMD ["./start.sh"]
