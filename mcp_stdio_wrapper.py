import sys
import json
import requests

FASTAPI_URL = "http://localhost:8000"

def handle_rpc(method, params, rpc_id):
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
        resp = requests.post(f"{FASTAPI_URL}/tools/list", json={})
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
            req = json.loads(line)
            method = req.get("method")
            params = req.get("params", {})
            rpc_id = req.get("id")
            result = handle_rpc(method, params, rpc_id)
            print(json.dumps(result), flush=True)
        except Exception as e:
            print(json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                "id": None
            }), flush=True)

if __name__ == "__main__":
    main()
