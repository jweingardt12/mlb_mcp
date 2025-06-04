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
        pass # Keep original rpc_id if coercion fails

    start_time = time.time()
    sys.stderr.write(f"INFO: [{rpc_id}] Handling method: {method} with params: {json.dumps(params)}\n")

    # Handle core MCP protocol methods directly in the wrapper for faster response
    if method == "exit":
        sys.stderr.write(f"INFO: [{rpc_id}] Received exit request, terminating\n")
        sys.exit(0)

    response_data = None
    try:
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": rpc_id
        }

        if method == "tools/list":
            try:
                sys.stderr.write(f"INFO: [{rpc_id}] Attempting GET {FASTAPI_URL}/tools\n")
                t_get_start = time.time()
                resp = requests.get(f"{FASTAPI_URL}/tools", timeout=10.0) # Increased timeout
                t_get_end = time.time()
                sys.stderr.write(f"INFO: [{rpc_id}] GET /tools completed in {t_get_end - t_get_start:.4f}s, status: {resp.status_code}\n")
                resp.raise_for_status() # Raise an exception for bad status codes
                data = resp.json()
                sys.stderr.write(f"INFO: [{rpc_id}] Got tools list from /tools endpoint: {json.dumps(data)}\n")
                response_data = {"jsonrpc": "2.0", "result": data, "id": rpc_id}
            except Exception as e:
                sys.stderr.write(f"WARN: [{rpc_id}] Failed to get tools from /tools (took {time.time() - t_get_start:.4f}s if started), falling back to JSON-RPC: {str(e)}\n")
                # Fall through to the JSON-RPC POST endpoint if response_data is still None
        
        if response_data is None: # If not tools/list or if GET /tools failed
            sys.stderr.write(f"INFO: [{rpc_id}] Sending to JSON-RPC endpoint (/mcp) for method '{method}': {json.dumps(jsonrpc_request)}\n")
            t_post_start = time.time()
            # Use a longer timeout for tools/list specifically if it falls back to POST
            post_timeout = 12.0 if method == "tools/list" else 5.0
            resp = requests.post(f"{FASTAPI_URL}/mcp", json=jsonrpc_request, timeout=post_timeout)
            t_post_end = time.time()
            sys.stderr.write(f"INFO: [{rpc_id}] POST /jsonrpc for method '{method}' completed in {t_post_end - t_post_start:.4f}s, status: {resp.status_code}\n")
            
            if resp.status_code == 200:
                response_data = resp.json()
            else:
                sys.stderr.write(f"ERROR: [{rpc_id}] JSON-RPC request failed with status {resp.status_code}: {resp.text}\n")
                response_data = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": f"HTTP error {resp.status_code}: {resp.text}"},
                    "id": rpc_id
                }

    except Exception as e:
        sys.stderr.write(f"ERROR: [{rpc_id}] Exception in handle_rpc: {str(e)}\n")
        response_data = {
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            "id": rpc_id
        }
    
    end_time = time.time()
    sys.stderr.write(f"INFO: [{rpc_id}] Total time for handle_rpc method '{method}': {end_time - start_time:.4f}s\n")
    return response_data

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
