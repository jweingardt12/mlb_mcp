"""
Ultra-minimal MCP server for Smithery testing.
Returns tools in the simplest possible format.
"""
from fastapi import FastAPI, Request
import json

app = FastAPI()

# Define tools in the simplest format
TOOLS = [
    {
        "name": "get_player_stats",
        "description": "Get player stats",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
    }
]

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/")
async def handle_request(request: Request):
    """Handle all MCP requests"""
    body = await request.json()
    method = body.get("method")
    request_id = body.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mlb_mcp", "version": "1.0.0"}
            },
            "id": request_id
        }
    
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "result": {"tools": TOOLS},
            "id": request_id
        }
    
    elif method == "tools/call":
        return {
            "jsonrpc": "2.0",
            "result": {
                "content": [{"type": "text", "text": "Test response"}]
            },
            "id": request_id
        }
    
    return {
        "jsonrpc": "2.0",
        "error": {"code": -32601, "message": "Method not found"},
        "id": request_id
    }