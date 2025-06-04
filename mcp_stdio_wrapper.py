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
    
    sys.stderr.write(f"INFO: Handling method: {method} with params: {json.dumps(params)}\n")
    
    # Handle core MCP protocol methods directly in the wrapper for faster response
    if method == "exit":
        sys.stderr.write("INFO: Received exit request, terminating\n")
        sys.exit(0)
    
    # For all other methods, use the unified JSON-RPC endpoint
    try:
        # Create a proper JSON-RPC request
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": rpc_id
        }
        
        # Special case for tools/list - try the lightweight endpoint first
        if method == "tools/list":
            try:
                # First try the lightweight GET endpoint for faster response
                resp = requests.get(f"{FASTAPI_URL}/tools", timeout=3.0)
                data = resp.json()
                sys.stderr.write(f"INFO: Got tools list from /tools endpoint: {json.dumps(data)}\n")
                return {"jsonrpc": "2.0", "result": data, "id": rpc_id}
            except Exception as e:
                sys.stderr.write(f"WARN: Failed to get tools from /tools, falling back to JSON-RPC: {str(e)}\n")
                # Fall through to the JSON-RPC endpoint
        
        # Send the request to the unified JSON-RPC endpoint
        sys.stderr.write(f"INFO: Sending to JSON-RPC endpoint: {json.dumps(jsonrpc_request)}\n")
        resp = requests.post(f"{FASTAPI_URL}/jsonrpc", json=jsonrpc_request, timeout=5.0)
        
        # Process the response
        if resp.status_code == 200:
            return resp.json()
        else:
            sys.stderr.write(f"ERROR: JSON-RPC request failed with status {resp.status_code}: {resp.text}\n")
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"HTTP error {resp.status_code}: {resp.text}"},
                "id": rpc_id
            }
    except Exception as e:
        sys.stderr.write(f"ERROR: Exception in handle_rpc: {str(e)}\n")
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            "id": rpc_id
        }
    # This code is unreachable - all cases are handled above
    # Keeping this comment as a placeholder

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
