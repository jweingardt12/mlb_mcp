"""
MCP server with dynamic tool listing for Smithery.
"""
from fastapi import FastAPI, Request
from datetime import datetime
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlb_mcp")

app = FastAPI()

# Tool definitions
TOOLS = [
    {
        "name": "get_player_stats",
        "description": "Get player statcast data by name",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Player name"},
                "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "End date YYYY-MM-DD"}
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
                "team": {"type": "string", "description": "Team name or abbreviation"},
                "year": {"type": "integer", "description": "Year/season"},
                "type": {"type": "string", "enum": ["batting", "pitching"]}
            },
            "required": ["team", "year", "type"]
        }
    },
    {
        "name": "get_leaderboard",
        "description": "Get leaderboard for a stat and season",
        "inputSchema": {
            "type": "object",
            "properties": {
                "stat": {"type": "string", "description": "Statistic (e.g., HR, AVG, ERA)"},
                "season": {"type": "integer", "description": "Season year"},
                "type": {"type": "string", "enum": ["batting", "pitching"]}
            },
            "required": ["stat", "season"]
        }
    },
    {
        "name": "statcast_leaderboard",
        "description": "Get Statcast leaderboard for date range",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "min_ev": {"type": "number"},
                "result": {"type": "string"},
                "order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"}
            },
            "required": ["start_date", "end_date"]
        }
    }
]

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/")
async def handle_mcp(request: Request):
    """Main MCP endpoint"""
    body = await request.json()
    method = body.get("method")
    params = body.get("params", {})
    request_id = body.get("id")
    
    logger.info(f"MCP request: {method}")
    
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
        tool = params.get("name")
        args = params.get("arguments", {})
        
        # Simple placeholder responses
        result_text = f"Called {tool} with {json.dumps(args)}"
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "content": [{"type": "text", "text": result_text}]
            },
            "id": request_id
        }
    
    else:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": f"Method not found: {method}"},
            "id": request_id
        }