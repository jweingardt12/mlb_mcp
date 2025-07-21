from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import json
import logging
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mlb_mcp_debug")

app = FastAPI()

# Simple test tool
TEST_TOOL = {
    "name": "echo",
    "description": "Echoes back the input message",
    "inputSchema": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Message to echo"
            }
        },
        "required": ["message"]
    }
}

# Log all requests middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"=== INCOMING REQUEST ===")
    logger.info(f"Method: {request.method}")
    logger.info(f"URL: {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    logger.info(f"Query params: {dict(request.query_params)}")
    
    # For POST requests, log the body
    if request.method == "POST":
        body = await request.body()
        request._body = body  # Store for later use
        try:
            body_json = json.loads(body) if body else None
            logger.info(f"Body: {json.dumps(body_json, indent=2)}")
        except:
            logger.info(f"Body (raw): {body}")
    
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "ok", "server": "mlb_mcp_debug", "time": datetime.now().isoformat()}

@app.get("/health")
async def health():
    logger.info("Health check called")
    return {"status": "healthy", "time": datetime.now().isoformat()}

# All possible MCP endpoints
@app.get("/mcp")
async def mcp_get(request: Request):
    logger.info("GET /mcp called")
    response = {
        "protocolVersion": "2025-03-26",
        "capabilities": {
            "tools": {"list": True, "call": True}
        },
        "tools": [TEST_TOOL]
    }
    logger.info(f"Returning: {json.dumps(response, indent=2)}")
    return response

@app.post("/mcp")
async def mcp_post(request: Request):
    logger.info("POST /mcp called")
    
    # Get body from stored value
    body = request._body if hasattr(request, '_body') else await request.body()
    data = json.loads(body) if body else {}
    
    method = data.get("method", "")
    params = data.get("params", {})
    rpc_id = data.get("id", 1)
    
    logger.info(f"JSON-RPC method: {method}")
    logger.info(f"Params: {params}")
    
    if method == "tools/list":
        response = {
            "jsonrpc": "2.0",
            "result": {"tools": [TEST_TOOL]},
            "id": rpc_id
        }
    elif method == "initialize":
        response = {
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
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        if tool_name == "echo":
            response = {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Echo: {arguments.get('message', 'no message')}"
                        }
                    ]
                },
                "id": rpc_id
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32602,
                    "message": f"Unknown tool: {tool_name}"
                },
                "id": rpc_id
            }
    else:
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            },
            "id": rpc_id
        }
    
    logger.info(f"Returning: {json.dumps(response, indent=2)}")
    return response

@app.delete("/mcp")
async def mcp_delete():
    logger.info("DELETE /mcp called")
    return {"status": "ok", "message": "Session terminated"}

# Alternative endpoints that might be called
@app.get("/tools")
async def tools_get():
    logger.info("GET /tools called")
    return {"tools": [TEST_TOOL]}

@app.post("/tools")
async def tools_post():
    logger.info("POST /tools called")
    return {"tools": [TEST_TOOL]}

@app.post("/tools/list")
async def tools_list_post(request: Request):
    logger.info("POST /tools/list called")
    body = request._body if hasattr(request, '_body') else await request.body()
    data = json.loads(body) if body else {}
    rpc_id = data.get("id", 1)
    
    response = {
        "jsonrpc": "2.0",
        "result": {"tools": [TEST_TOOL]},
        "id": rpc_id
    }
    logger.info(f"Returning: {json.dumps(response, indent=2)}")
    return response

@app.post("/initialize")
async def initialize_post(request: Request):
    logger.info("POST /initialize called")
    body = request._body if hasattr(request, '_body') else await request.body()
    data = json.loads(body) if body else {}
    rpc_id = data.get("id", 1)
    
    response = {
        "jsonrpc": "2.0",
        "result": {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {"call": True, "list": True}
            }
        },
        "id": rpc_id
    }
    logger.info(f"Returning: {json.dumps(response, indent=2)}")
    return response

# Log server startup
logger.info(f"Starting MCP debug server on port {os.environ.get('PORT', 8000)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)