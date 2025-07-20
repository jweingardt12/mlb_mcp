from fastapi import FastAPI, Request, HTTPException
from typing import Optional, Dict, Any, List
from datetime import datetime
import os
import json
from fastapi.responses import JSONResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlb_mcp")

# Configure FastAPI
app = FastAPI(
    title="MLB Stats MCP",
    description="MCP server exposing MLB/Fangraphs data via pybaseballstats.",
    docs_url="/docs",
    redoc_url=None
)

@app.on_event("startup")
async def startup_event():
    """Initialize logging and prepare for optimal performance"""
    logger.info("MLB Stats MCP server starting up...")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"PORT environment variable: {os.environ.get('PORT', 'Not set')}")
    logger.info("Server ready - tools available via MCP protocol")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    logger.info("MLB Stats MCP server shutting down...")

# Health check endpoints
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MLB Stats MCP",
        "timestamp": datetime.now().isoformat()
    }

# Define the tools with proper MCP structure
TOOLS = [
    {
        "name": "get_player_stats",
        "description": "Get player statcast data by name (optionally filter by date range)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Player name to search for"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "get_team_stats",
        "description": "Get team stats for a given team and year",
        "inputSchema": {
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "Team name or abbreviation"
                },
                "year": {
                    "type": "integer",
                    "description": "Year/season to get stats for"
                },
                "type": {
                    "type": "string",
                    "description": "Type of stats (batting or pitching)",
                    "enum": ["batting", "pitching"]
                }
            },
            "required": ["team", "year", "type"]
        }
    },
    {
        "name": "get_leaderboard",
        "description": "Get leaderboard for a given stat and season",
        "inputSchema": {
            "type": "object",
            "properties": {
                "stat": {
                    "type": "string",
                    "description": "Statistic to get leaderboard for (e.g., 'HR', 'AVG', 'ERA')"
                },
                "season": {
                    "type": "integer",
                    "description": "Season year to get leaderboard for"
                },
                "type": {
                    "type": "string",
                    "description": "Type of leaderboard (batting or pitching)",
                    "enum": ["batting", "pitching"]
                }
            },
            "required": ["stat", "season"]
        }
    },
    {
        "name": "statcast_leaderboard",
        "description": "Get event-level Statcast leaderboard for a date range, filtered by result (e.g., home run) and sorted by exit velocity, etc.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return"
                },
                "min_ev": {
                    "type": "number",
                    "description": "Minimum exit velocity (optional)"
                },
                "result": {
                    "type": "string",
                    "description": "Filter by result type, e.g., 'Home Run' (optional)"
                },
                "order": {
                    "type": "string",
                    "description": "Sort order: 'asc' or 'desc' (optional, default desc)"
                }
            },
            "required": ["start_date", "end_date"]
        }
    }
]

# MCP Protocol Implementation
@app.post("/mcp/v1/init")
async def mcp_init(request: Request):
    """MCP initialization endpoint"""
    try:
        data = await request.json()
        logger.info(f"MCP init request: {json.dumps(data)}")
        
        return {
            "protocolVersion": "2024-11-08",
            "capabilities": {
                "tools": True
            }
        }
    except Exception as e:
        logger.error(f"Error in MCP init: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/v1/tools/list")
async def mcp_tools_list(request: Request):
    """MCP tools listing endpoint"""
    try:
        data = await request.json()
        logger.info(f"MCP tools/list request: {json.dumps(data)}")
        
        return {
            "tools": TOOLS
        }
    except Exception as e:
        logger.error(f"Error in MCP tools/list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/v1/tools/call")
async def mcp_tools_call(request: Request):
    """MCP tool execution endpoint"""
    try:
        data = await request.json()
        logger.info(f"MCP tools/call request: {json.dumps(data)}")
        
        tool_name = data.get("name")
        arguments = data.get("arguments", {})
        
        # For now, return a placeholder response
        # In production, this would execute the actual tool
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Tool '{tool_name}' called with arguments: {json.dumps(arguments)}. Full implementation coming soon."
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error in MCP tools/call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy JSON-RPC endpoints for compatibility
@app.post("/mcp")
async def jsonrpc_endpoint(request: Request):
    """Generic JSON-RPC endpoint for MCP operations"""
    try:
        data = await request.json()
        method = data.get("method")
        params = data.get("params", {})
        rpc_id = data.get("id", 1)
        
        logger.info(f"JSON-RPC request: {method}")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-08",
                    "capabilities": {
                        "tools": {
                            "list": True,
                            "call": True
                        }
                    },
                    "serverInfo": {
                        "name": "MLB Stats MCP",
                        "version": "1.0.0"
                    }
                },
                "id": rpc_id
            }
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "result": {
                    "tools": TOOLS
                },
                "id": rpc_id
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool '{tool_name}' called with arguments: {json.dumps(arguments)}. Full implementation coming soon."
                        }
                    ]
                },
                "id": rpc_id
            }
        else:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                },
                "id": rpc_id
            }
    except Exception as e:
        logger.error(f"Error in JSON-RPC endpoint: {str(e)}")
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": str(e)
            },
            "id": data.get("id", 1) if "data" in locals() else 1
        }

# Additional endpoints for debugging
@app.get("/mcp/tools")
async def get_tools():
    """GET endpoint to verify tools are configured"""
    return {"tools": TOOLS}

@app.get("/mcp/info")
async def get_info():
    """GET endpoint for server information"""
    return {
        "name": "MLB Stats MCP",
        "version": "1.0.0",
        "protocol": "MCP 2024-11-08",
        "tools_count": len(TOOLS),
        "status": "operational"
    }