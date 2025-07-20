from fastapi import FastAPI, Request, HTTPException, Query
from typing import Optional, Dict, Any
from datetime import datetime
import os
from fastapi.responses import JSONResponse
import traceback

# Configure FastAPI with minimal startup operations
app = FastAPI(
    title="MLB Stats MCP",
    description="MCP server exposing MLB/Fangraphs data via pybaseballstats.",
    docs_url="/docs",
    redoc_url=None
)

@app.on_event("startup")
async def startup_event():
    """Initialize logging and prepare for optimal performance"""
    import logging
    import sys
    logger = logging.getLogger("uvicorn")
    logger.info("MLB Stats MCP server starting up...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PORT environment variable: {os.environ.get('PORT', 'Not set')}")
    logger.info("Startup complete - pybaseballstats will be loaded on first use")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    import logging
    logger = logging.getLogger("uvicorn")
    logger.info("MLB Stats MCP server shutting down...")

# Health check endpoint
@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": "MLB Stats MCP server is running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "service": "MLB Stats MCP",
        "timestamp": datetime.now().isoformat(),
        "dependencies": {
            "pybaseball_loaded": False,
            "note": "Heavy dependencies not loaded to reduce memory usage"
        }
    }

# Define the tools statically
STATIC_TOOLS = [
    {
        "name": "get_player_stats",
        "description": "Get player statcast data by name (optionally filter by date range: YYYY-MM-DD)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Player name to search for"},
                "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "get_team_stats",
        "description": "Get team stats for a given team and year. Type can be 'batting' or 'pitching'",
        "inputSchema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "description": "Team name or abbreviation"},
                "year": {"type": "integer", "description": "Year/season to get stats for"},
                "type": {"type": "string", "description": "Type of stats (batting or pitching)", "enum": ["batting", "pitching"]}
            },
            "required": ["team", "year"]
        }
    },
    {
        "name": "get_leaderboard",
        "description": "Get leaderboard for a given stat and season",
        "inputSchema": {
            "type": "object",
            "properties": {
                "stat": {"type": "string", "description": "Statistic to get leaderboard for (e.g., 'HR', 'AVG', 'ERA')"},
                "season": {"type": "integer", "description": "Season year to get leaderboard for"},
                "type": {"type": "string", "description": "Type of leaderboard (batting or pitching)", "enum": ["batting", "pitching"]}
            },
            "required": ["stat", "season"]
        }
    },
    {
        "name": "statcast_leaderboard",
        "description": "Get event-level Statcast leaderboard for a date range",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
                "limit": {"type": "integer", "description": "Number of results to return"},
                "min_ev": {"type": "number", "description": "Minimum exit velocity (optional)"},
                "result": {"type": "string", "description": "Filter by result type, e.g., 'Home Run' (optional)"},
                "order": {"type": "string", "description": "Sort order: 'asc' or 'desc' (optional, default desc)"}
            },
            "required": ["start_date", "end_date"]
        }
    }
]

# Tool listing endpoints
@app.get("/tools")
async def tools_list_get():
    """Lightweight endpoint for tool discovery"""
    return {"protocolVersion": "2025-03-26", "tools": STATIC_TOOLS}

@app.get("/mcp")
async def mcp_get_handler():
    """Handles GET requests to the /mcp endpoint"""
    return {"protocolVersion": "2025-03-26", "tools": STATIC_TOOLS}

@app.post("/mcp")
async def jsonrpc_endpoint(request: Request):
    """Generic JSON-RPC endpoint for all MCP operations"""
    try:
        data = await request.json()
        method = data.get("method")
        params = data.get("params", {})
        rpc_id = data.get("id", 1)
        
        print(f"Received JSON-RPC request: {method} with ID {rpc_id}")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {
                        "tools": {
                            "call": True,
                            "list": True
                        }
                    },
                    "serverInfo": {
                        "name": "MLB Stats MCP",
                        "version": "1.0.0"
                    }
                },
                "id": rpc_id
            }
        elif method == "shutdown":
            return {"jsonrpc": "2.0", "result": None, "id": rpc_id}
        elif method == "ping":
            return {"jsonrpc": "2.0", "result": {}, "id": rpc_id}
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "result": {"protocolVersion": "2025-03-26", "tools": STATIC_TOOLS},
                "id": rpc_id
            }
        elif method == "tools/call":
            # Return error for now - pybaseballstats not loaded
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": "Tool execution temporarily disabled to reduce memory usage"
                },
                "id": rpc_id
            }
        else:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": rpc_id
            }
    except Exception as e:
        print(f"Error in JSON-RPC endpoint: {str(e)}")
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": str(e)},
            "id": data.get("id", 1) if "data" in locals() else 1
        }

@app.post("/tools/list")
async def mcp_tools_list():
    """Standard MCP endpoint for tool listing"""
    return {
        "jsonrpc": "2.0",
        "result": {"tools": STATIC_TOOLS},
        "id": 1
    }

@app.post("/initialize")
async def mcp_initialize(request: Request):
    """Handle MCP initialization requests"""
    try:
        data = await request.json()
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
                    "name": "MLB Stats MCP",
                    "version": "1.0.0"
                }
            },
            "id": data.get("id", 1)
        }
    except Exception as e:
        print(f"Error in initialize: {str(e)}")
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": str(e)},
            "id": 1
        }