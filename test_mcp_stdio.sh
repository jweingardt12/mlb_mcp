#!/bin/bash
# Start FastAPI in the background with logs to STDERR
uvicorn main:app --host 0.0.0.0 --port 8000 --log-config uvicorn_log_config.json &
FASTAPI_PID=$!

# Wait for FastAPI to be ready
sleep 2

echo 'Testing MCP STDIO Wrapper...'

# Test initialize
INIT_RESP=$(echo '{"jsonrpc": "2.0", "method": "initialize", "id": 1}' | python3 mcp_stdio_wrapper.py)
echo "\n[initialize response]:"
echo "$INIT_RESP"

# Test tools/list
TOOLS_RESP=$(echo '{"jsonrpc": "2.0", "method": "tools/list", "id": 2}' | python3 mcp_stdio_wrapper.py)
echo "\n[tools/list response]:"
echo "$TOOLS_RESP"

# Kill FastAPI
kill $FASTAPI_PID
