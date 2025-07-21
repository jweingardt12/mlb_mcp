from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os

app = FastAPI()

# Single test tool
TEST_TOOL = {
    "name": "test_tool",
    "description": "A simple test tool",
    "inputSchema": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Test message"
            }
        },
        "required": ["message"]
    }
}

@app.get("/")
async def root():
    return {"status": "ok", "server": "mlb_mcp_minimal"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# MCP endpoints
@app.get("/mcp")
async def mcp_get(request: Request):
    """GET /mcp for tool discovery"""
    return {
        "protocolVersion": "2025-03-26",
        "tools": [TEST_TOOL]
    }

@app.post("/mcp")
async def mcp_post(request: Request):
    """POST /mcp for JSON-RPC"""
    data = await request.json()
    method = data.get("method", "")
    rpc_id = data.get("id", 1)
    
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "result": {"tools": [TEST_TOOL]},
            "id": rpc_id
        }
    elif method == "initialize":
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {
                    "tools": {"call": True, "list": True}
                }
            },
            "id": rpc_id
        }
    elif method == "tools/call":
        params = data.get("params", {})
        tool_name = params.get("name", "")
        
        if tool_name == "test_tool":
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Test response: {params.get('arguments', {}).get('message', 'no message')}"
                        }
                    ]
                },
                "id": rpc_id
            }
    
    return {
        "jsonrpc": "2.0",
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}"
        },
        "id": rpc_id
    }

@app.delete("/mcp")
async def mcp_delete():
    return {"status": "ok"}

# Alternative endpoints
@app.post("/tools/list")
async def tools_list_post():
    return {
        "jsonrpc": "2.0", 
        "result": {"tools": [TEST_TOOL]},
        "id": 1
    }

@app.get("/tools")
async def tools_get():
    return {"tools": [TEST_TOOL]}

@app.post("/tools")
async def tools_post():
    return {"tools": [TEST_TOOL]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)