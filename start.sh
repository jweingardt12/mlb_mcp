#!/bin/sh
set -e

# Log startup information
echo "Starting MLB Stats MCP server..."
echo "PORT: ${PORT:-8000}"
echo "Python version: $(python --version)"

# Ensure PORT is set
export PORT=${PORT:-8000}

# Start uvicorn with proper settings
exec python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --log-config uvicorn_log_config.yaml \
    --timeout-keep-alive 5 \
    --limit-concurrency 100 \
    --no-access-log