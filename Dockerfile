# TEMPORARY DEBUG DOCKERFILE - Simple single-stage for debugging
FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks and minimal Python dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir fastapi==0.116.1 uvicorn[standard]==0.35.0

# Copy debug server
COPY main_debug_logging.py main.py

# EXPOSE is informational; Smithery uses the $PORT env var
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Set Python optimization flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run with detailed logging
CMD sh -c "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"