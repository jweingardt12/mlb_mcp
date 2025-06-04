import sys
import json
import requests
import time

FASTAPI_URL = "http://localhost:8000"

# Wait for FastAPI to be ready
# Allow the FastAPI server extra time to start up in case package imports are slow
# Smithery will wait for the wrapper to respond on STDIO, so we loop up to 60
# times (about 30 seconds) before giving up.
sys.stderr.write("INFO: Waiting for FastAPI server to start...\n")
for _ in range(60):  # Try for up to 30 seconds
    try:
        # Use the root endpoint instead of /docs since we disabled the docs endpoints
        r = requests.get(f"{FASTAPI_URL}/", timeout=0.5)
        if r.status_code == 200:
            sys.stderr.write(f"INFO: FastAPI server is ready: {r.text}\n")
            break
    except Exception as e:
        sys.stderr.write(f"INFO: Waiting for server... ({str(e)})\n")
    time.sleep(0.5)
else:
    sys.stderr.write("ERROR: FastAPI server did not start in time.\n")
    sys.exit(1)

def handle_rpc(method, params, rpc_id):
    # Coerce id to int if possible, for Smithery compatibility
    try:
        if rpc_id is not None:
            rpc_id = int(rpc_id)
    except Exception:
        pass
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "result": {
                "capabilities": {
                    "tools": {
                        "call": True,
                        "list": True
                    }
                },
                "serverInfo": {
                    "name": "Pybaseball MCP Server",
                    "version": "1.0.0"
                }
            },
            "id": rpc_id
        }
    elif method == "shutdown":
        return {"jsonrpc": "2.0", "result": None, "id": rpc_id}
    elif method == "exit":
        sys.exit(0)
    elif method == "get_player_stats":
        resp = requests.get(f"{FASTAPI_URL}/player", params=params)
    elif method == "get_team_stats":
        resp = requests.get(f"{FASTAPI_URL}/team_stats", params=params)
    elif method == "get_leaderboard":
        resp = requests.get(f"{FASTAPI_URL}/leaderboard", params=params)
    elif method == "tools/list":
        try:
            # First try the lightweight GET endpoint for faster response
            resp = requests.get(f"{FASTAPI_URL}/tools", timeout=1.0)
            data = resp.json()
            sys.stderr.write(f"INFO: Got tools list from /tools endpoint: {json.dumps(data)}\n")
            return {"jsonrpc": "2.0", "result": data, "id": rpc_id}
        except Exception as e:
            sys.stderr.write(f"WARN: Failed to get tools from /tools, falling back to /tools/list: {str(e)}\n")
            # Fall back to the standard JSON-RPC endpoint if needed
            resp = requests.post(f"{FASTAPI_URL}/tools/list", json={})
            data = resp.json()
            # If the FastAPI endpoint returns a JSON-RPC response, extract the 'result' field
            if "result" in data:
                return {"jsonrpc": "2.0", "result": data["result"], "id": rpc_id}
            else:
                return {"jsonrpc": "2.0", "result": data, "id": rpc_id}
    else:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
            "id": rpc_id
        }
    try:
        data = resp.json()
        return {"jsonrpc": "2.0", "result": data, "id": rpc_id}
    except Exception as e:
        return {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": rpc_id}

def main():
    for line in sys.stdin:
        try:
            sys.stderr.write(f"IN: {line}\n")
            req = json.loads(line)
            method = req.get("method")
            params = req.get("params", {})
            rpc_id = req.get("id")
            result = handle_rpc(method, params, rpc_id)
            sys.stderr.write(f"OUT: {json.dumps(result)}\n")
            print(json.dumps(result), flush=True)
        except Exception as e:
            error_obj = {
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                "id": None
            }
            sys.stderr.write(f"ERR: {json.dumps(error_obj)}\n")
            print(json.dumps(error_obj), flush=True)

if __name__ == "__main__":
    main()
