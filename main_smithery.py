"""
Minimal MCP server for Smithery deployment.
Tools are defined in smithery.yaml, so we just need to handle execution.
"""
from fastapi import FastAPI, Request, HTTPException
from datetime import datetime
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlb_mcp")

app = FastAPI(title="MLB Stats MCP")

@app.on_event("startup")
async def startup_event():
    logger.info("MLB Stats MCP server starting...")
    logger.info(f"PORT: {os.environ.get('PORT', 'Not set')}")

# Health check
@app.get("/")
async def root():
    return {"status": "ok", "message": "MLB Stats MCP server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# MCP JSON-RPC endpoint
@app.post("/")
async def handle_jsonrpc(request: Request):
    """Main JSON-RPC endpoint for MCP protocol"""
    try:
        body = await request.json()
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")
        
        logger.info(f"Received: method={method}, id={request_id}")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "mlb_mcp",
                        "version": "1.0.0"
                    }
                },
                "id": request_id
            }
        
        elif method == "tools/list":
            # Return empty list - tools are defined in smithery.yaml
            return {
                "jsonrpc": "2.0",
                "result": {
                    "tools": []
                },
                "id": request_id
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            logger.info(f"Tool call: {tool_name} with args: {arguments}")
            
            # Simple responses for each tool
            if tool_name == "get_player_stats":
                player_name = arguments.get("name", "Unknown")
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Player stats for {player_name}: Implementation coming soon. This tool will return batting statistics from MLB/Fangraphs."
                            }
                        ]
                    },
                    "id": request_id
                }
            
            elif tool_name == "get_team_stats":
                team = arguments.get("team", "Unknown")
                year = arguments.get("year", 2024)
                stat_type = arguments.get("type", "batting")
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Team {stat_type} stats for {team} in {year}: Implementation coming soon."
                            }
                        ]
                    },
                    "id": request_id
                }
            
            elif tool_name == "get_leaderboard":
                stat = arguments.get("stat", "HR")
                season = arguments.get("season", 2024)
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Leaderboard for {stat} in {season}: Implementation coming soon."
                            }
                        ]
                    },
                    "id": request_id
                }
            
            elif tool_name == "statcast_leaderboard":
                start_date = arguments.get("start_date")
                end_date = arguments.get("end_date")
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Statcast leaderboard from {start_date} to {end_date}: Implementation coming soon."
                            }
                        ]
                    },
                    "id": request_id
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    },
                    "id": request_id
                }
        
        else:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                },
                "id": request_id
            }
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": str(e)
            },
            "id": body.get("id") if "body" in locals() else None
        }