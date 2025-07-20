#!/bin/sh
set -e

# Log startup information
echo "Starting MLB Stats MCP server (minimal mode)..."
echo "PORT: ${PORT:-8000}"

# Ensure PORT is set
export PORT=${PORT:-8000}

# Start uvicorn with minimal settings
exec python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 5 \
    --limit-concurrency 50 \
    --loop uvloop