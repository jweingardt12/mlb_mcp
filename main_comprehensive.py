from fastapi import FastAPI, Request, HTTPException
from typing import Optional, Dict, Any, List
import os
import time
import asyncio
from datetime import datetime
from fastapi.responses import JSONResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# Create FastAPI app with minimal overhead
app = FastAPI(
    title="MLB Stats MCP",
    description="MCP server exposing MLB/Fangraphs data via pybaseballstats",
    docs_url="/docs",
    redoc_url=None
)

# Define tools statically to ensure immediate availability
TOOLS = [
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
                "season": {"type": "integer", "description": "Season year"},
                "type": {"type": "string", "description": "Type of leaderboard (batting or pitching)", "enum": ["batting", "pitching"]},
                "limit": {"type": "integer", "description": "Number of results to return", "default": 10}
            },
            "required": ["stat", "season"]
        }
    },
    {
        "name": "mlb_video_search",
        "description": "Search MLB video highlights using the MLB Film Room API",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "MLB Film Room search query"},
                "limit": {"type": "integer", "description": "Number of results", "default": 10},
                "page": {"type": "integer", "description": "Page number", "default": 0}
            },
            "required": ["query"]
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
                "limit": {"type": "integer", "description": "Number of results", "default": 10},
                "min_ev": {"type": "number", "description": "Minimum exit velocity"},
                "result": {"type": "string", "description": "Filter by result type"}
            },
            "required": ["start_date", "end_date"]
        }
    }
]

@app.on_event("startup")
async def startup_event():
    """Initialize server on startup"""
    logger.info("MLB Stats MCP server starting up...")
    logger.info(f"PORT: {os.environ.get('PORT', 'Not set')}")
    logger.info(f"Available tools: {[t['name'] for t in TOOLS]}")
    logger.info("Startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    logger.info("MLB Stats MCP server shutting down...")

# Health check endpoints
@app.get("/")
async def read_root():
    return {
        "status": "ok",
        "message": "MLB Stats MCP server is running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "MLB Stats MCP",
        "timestamp": datetime.now().isoformat()
    }

# MCP Protocol Endpoints - Multiple variations for compatibility
@app.get("/tools")
async def tools_list_get():
    """GET endpoint for tool discovery"""
    return {"tools": TOOLS}

@app.post("/tools")
async def tools_list_post():
    """POST endpoint for tool discovery"""
    return {"tools": TOOLS}

@app.get("/mcp/tools")
async def mcp_tools_get():
    """MCP tools endpoint (GET)"""
    return {"tools": TOOLS}

@app.post("/mcp/tools") 
async def mcp_tools_post():
    """MCP tools endpoint (POST)"""
    return {"tools": TOOLS}

@app.post("/tools/list")
async def tools_list():
    """Standard MCP tools/list endpoint"""
    return {
        "jsonrpc": "2.0",
        "result": {"tools": TOOLS},
        "id": 1
    }

# Main MCP endpoint
@app.get("/mcp")
async def mcp_get():
    """GET handler for /mcp endpoint"""
    return {"tools": TOOLS}

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Main JSON-RPC endpoint for MCP protocol"""
    try:
        data = await request.json()
        method = data.get("method")
        params = data.get("params", {})
        rpc_id = data.get("id", 1)
        
        logger.info(f"MCP request: method={method}, id={rpc_id}")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {
                        "tools": {"call": True, "list": True}
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
                "result": {"tools": TOOLS},
                "id": rpc_id
            }
        
        elif method == "tools/call":
            tool_name = params.get("name", "").replace("mlb-mcp-", "")
            tool_params = params.get("parameters") or params.get("arguments", {})
            
            logger.info(f"Tool call: {tool_name} with params: {tool_params}")
            
            # Execute tool call
            if tool_name == "get_player_stats":
                result = await get_player_stats_impl(**tool_params)
            elif tool_name == "get_team_stats":
                result = await get_team_stats_impl(**tool_params)
            elif tool_name == "get_leaderboard":
                result = await get_leaderboard_impl(**tool_params)
            elif tool_name == "mlb_video_search":
                result = await mlb_video_search_impl(**tool_params)
            elif tool_name == "statcast_leaderboard":
                result = await statcast_leaderboard_impl(**tool_params)
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
                    "id": rpc_id
                }
            
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": rpc_id
            }
        
        elif method == "ping":
            return {"jsonrpc": "2.0", "result": {}, "id": rpc_id}
        
        elif method == "shutdown":
            return {"jsonrpc": "2.0", "result": None, "id": rpc_id}
        
        else:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": rpc_id
            }
            
    except Exception as e:
        logger.error(f"Error in MCP endpoint: {str(e)}")
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": str(e)},
            "id": data.get("id", 1) if "data" in locals() else 1
        }

# Tool implementations with minimal dependencies
async def get_player_stats_impl(name: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Minimal implementation that defers heavy imports"""
    try:
        # Defer import until actually needed
        import pybaseballstats as pb
        from datetime import datetime, timedelta
        
        first, last = name.split()[0], " ".join(name.split()[1:])
        player_df = pb.retrosheet.player_lookup(first_name=first, last_name=last, return_pandas=True)
        
        if player_df.empty:
            return {"error": f"Player not found: {name}"}
        
        player_id = int(player_df.iloc[0]['key_mlbam'])
        
        # Get stats for date range
        if not start_date or not end_date:
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        stats = pb.statcast.statcast_date_range_pitch_by_pitch(start_date, end_date, return_pandas=True)
        stats = stats[stats['batter'] == player_id]
        
        return {
            "player": name,
            "player_id": player_id,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(stats),
            "stats": stats.head(100).to_dict(orient="records") if not stats.empty else []
        }
    except Exception as e:
        logger.error(f"Error in get_player_stats: {str(e)}")
        return {"error": str(e)}

async def get_team_stats_impl(team: str, year: int, type: str = "batting"):
    """Minimal team stats implementation"""
    try:
        import pybaseballstats as pb
        from pybaseballstats.bref_teams import BREFTeams
        
        # Simple team mapping
        team_upper = team.replace(" ", "_").upper()
        
        # Try to find the team
        bref_team = None
        for t in BREFTeams:
            if t.name == team_upper or t.value.upper() == team_upper:
                bref_team = t
                break
        
        if not bref_team:
            return {"error": f"Team not found: {team}"}
        
        if type == "batting":
            stats = pb.bref_teams.team_standard_batting(bref_team, year, return_pandas=True)
        else:
            # For pitching, return placeholder
            return {
                "team": team,
                "year": year,
                "type": type,
                "stats": [{"note": "Pitching stats require full implementation"}]
            }
        
        return {
            "team": team,
            "year": year,
            "type": type,
            "count": len(stats),
            "stats": stats.to_dict(orient="records") if not stats.empty else []
        }
    except Exception as e:
        logger.error(f"Error in get_team_stats: {str(e)}")
        return {"error": str(e)}

async def get_leaderboard_impl(stat: str, season: int, type: str = "batting", limit: int = 10):
    """Minimal leaderboard implementation"""
    try:
        # For minimal implementation, return placeholder data
        return {
            "stat": stat,
            "season": season,
            "type": type,
            "content": [
                {
                    "player_name": f"Player {i+1}",
                    "team": "TBD",
                    stat: f"{100-i*5}"
                }
                for i in range(min(limit, 5))
            ]
        }
    except Exception as e:
        logger.error(f"Error in get_leaderboard: {str(e)}")
        return {"error": str(e)}

async def mlb_video_search_impl(query: str, limit: int = 10, page: int = 0):
    """MLB video search implementation"""
    try:
        import requests
        
        url = "https://fastball-gateway.mlb.com/graphql"
        headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "origin": "https://www.mlb.com",
            "referer": "https://www.mlb.com/video/"
        }
        
        body = {
            "query": """query Search($query: String!, $page: Int, $limit: Int) { 
                search(query: $query, limit: $limit, page: $page) { 
                    plays { 
                        mediaPlayback { 
                            id slug title description date 
                        } 
                    } 
                } 
            }""",
            "variables": {
                "query": query,
                "limit": limit,
                "page": page
            }
        }
        
        response = requests.post(url, headers=headers, json=body, timeout=10)
        if response.status_code == 200:
            data = response.json()
            plays = data.get("data", {}).get("search", {}).get("plays", [])
            
            videos = []
            for play in plays[:limit]:
                for mp in play.get("mediaPlayback", []):
                    videos.append({
                        "type": "resource",
                        "resource": {
                            "id": mp.get("id"),
                            "title": mp.get("title"),
                            "description": mp.get("description"),
                            "date": mp.get("date"),
                            "uri": f"https://www.mlb.com/video/{mp.get('slug')}" if mp.get('slug') else None
                        }
                    })
            
            return {"content": videos}
        else:
            return {"content": [], "error": f"API returned {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Error in mlb_video_search: {str(e)}")
        return {"content": [], "error": str(e)}

async def statcast_leaderboard_impl(start_date: str, end_date: str, limit: int = 10, 
                                   min_ev: Optional[float] = None, result: Optional[str] = None):
    """Minimal statcast leaderboard implementation"""
    try:
        # Return placeholder data for minimal implementation
        return {
            "content": [
                {
                    "batter_name": f"Player {i+1}",
                    "exit_velocity": 115.0 - i,
                    "date": start_date,
                    "result": result or "Home Run"
                }
                for i in range(min(limit, 5))
            ]
        }
    except Exception as e:
        logger.error(f"Error in statcast_leaderboard: {str(e)}")
        return {"content": [], "error": str(e)}

# Additional compatibility endpoints
@app.post("/initialize")
async def initialize(request: Request):
    """Direct initialize endpoint"""
    return {
        "jsonrpc": "2.0",
        "result": {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {"call": True, "list": True}
            },
            "serverInfo": {
                "name": "MLB Stats MCP",
                "version": "1.0.0"
            }
        },
        "id": 1
    }

@app.post("/tools/call")
async def tools_call(request: Request):
    """Direct tools/call endpoint"""
    data = await request.json()
    data["method"] = "tools/call"
    return await mcp_endpoint(request)

# API documentation endpoint
@app.get("/api/tools")
async def api_tools():
    """REST API endpoint for tool discovery"""
    return {"tools": TOOLS}