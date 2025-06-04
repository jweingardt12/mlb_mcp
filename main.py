from fastapi import FastAPI, HTTPException, Request
from typing import Optional, Dict, Any, List
from datetime import datetime
import importlib
import asyncio
import functools
import time

# Lazy imports - don't import pybaseball until needed
# This prevents slow startup times that can cause timeouts
pybaseball = None

# Configure FastAPI with minimal startup operations
app = FastAPI(
    title="MLB Stats MCP",
    description="MCP server exposing MLB/Fangraphs data via pybaseball.",
    # Keep docs enabled but with minimal overhead
    docs_url="/docs",
    redoc_url=None
)

# Simple health check endpoint
@app.get("/")
def read_root():
    return {"status": "ok", "message": "MLB Stats MCP server is running"}

# Lazy loading helper function
def load_pybaseball():
    global pybaseball
    if pybaseball is None:
        import pybaseball as pb
        pybaseball = pb
    return pybaseball

# Timeout decorator for functions
def with_timeout(timeout_seconds=10):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Convert the function to a coroutine if it's not already one
                if asyncio.iscoroutinefunction(func):
                    coro = func(*args, **kwargs)
                else:
                    coro = asyncio.to_thread(func, *args, **kwargs)
                
                # Execute with timeout
                return await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Operation timed out")
        return wrapper
    return decorator

def call_tool(tool_name, params):
    # Load pybaseball only when a tool is actually called
    pb = load_pybaseball() # Assuming load_pybaseball() is robust or its errors can propagate

    try:
        if tool_name == "get_player_stats":
            return get_player_stats(**params)
        elif tool_name == "get_team_stats":
            return get_team_stats(**params)
        elif tool_name == "get_leaderboard":
            return get_leaderboard(**params)
        else:
            # This case should ideally be caught by the checks in jsonrpc_endpoint
            # before calling call_tool, but as a safeguard:
            raise ValueError(f"Unknown tool: {tool_name}")
    except Exception as e:
        error_message = f"Error executing tool '{tool_name}' with params '{params}': {str(e)}"
        print(error_message) # Log to server console (Smithery logs)
        # Re-raise the exception so it can be caught by the jsonrpc_endpoint's main error handler
        # and returned as a proper JSON-RPC error response to the client.
        raise e

# Define the tools statically to avoid any computation during tool listing
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
            "required": ["team", "year", "type"]
        }
    },
    {
        "name": "get_leaderboard",
        "description": "Get leaderboard for a given stat and season. Type can be 'batting' or 'pitching'",
        "inputSchema": {
            "type": "object",
            "properties": {
                "stat": {"type": "string", "description": "Statistic to get leaderboard for (e.g., 'HR', 'AVG', 'ERA')"},
                "season": {"type": "integer", "description": "Season year to get leaderboard for"},
                "type": {"type": "string", "description": "Type of leaderboard (batting or pitching)", "enum": ["batting", "pitching"]}
            },
            "required": ["stat", "season"]
        }
    }
]

# Create a dedicated endpoint just for tool listing that responds instantly
@app.get("/tools")
async def tools_list_get():
    """Lightweight endpoint for tool discovery that responds instantly"""
    return {"protocolVersion": "2025-03-26", "tools": STATIC_TOOLS}

@app.get("/mcp")
async def mcp_get_handler():
    """Handles GET requests to the /mcp endpoint, primarily for tool discovery."""
    # Smithery's technical requirements state /mcp must handle GET.
    # Returning the tools list here provides another discovery mechanism.
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
                    "protocolVersion": "2025-03-26",  # Added protocol version here
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
            return {"jsonrpc": "2.0", "result": {}, "id": rpc_id}  # Simple pong with empty object
        elif method == "tools/call":
            # Handle gateway-style tool calls
            actual_tool_name = params.get("name")
            actual_tool_params = params.get("parameters", {})
            if actual_tool_name in ["get_player_stats", "get_team_stats", "get_leaderboard"]:
                result = call_tool(actual_tool_name, actual_tool_params)
                return {"jsonrpc": "2.0", "result": result, "id": rpc_id}
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Tool not found within tools/call: {actual_tool_name}"},
                    "id": rpc_id
                }
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "result": {"protocolVersion": "2025-03-26", "tools": STATIC_TOOLS},
                "id": rpc_id
            }
        elif method in ["get_player_stats", "get_team_stats", "get_leaderboard"]:
            result = call_tool(method, params)
            return {"jsonrpc": "2.0", "result": result, "id": rpc_id}
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
    # Return the static tool list immediately without any computation
    return {
        "jsonrpc": "2.0",
        "result": {"tools": STATIC_TOOLS},
        "id": 1
    }

@app.post("/initialize")
async def mcp_initialize(request: Request):
    """Handle MCP initialization requests directly"""
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

@app.post("/tools/call")
async def mcp_tools_call(request: Request):
    data = await request.json()
    try:
        method = data.get("method")
        params = data.get("params", {})
        
        # Handle MCP protocol methods directly
        if method == "initialize":
            return await mcp_initialize(request)
        elif method == "shutdown":
            return {"jsonrpc": "2.0", "result": None, "id": data.get("id", 1)}
        
        # Validate method before attempting to call
        if method not in ["get_player_stats", "get_team_stats", "get_leaderboard"]:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                },
                "id": data.get("id", 1)
            }
        
        # Call the tool with a timeout to prevent hanging
        result = call_tool(method, params)
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": data.get("id", 1)
        }
    except Exception as e:
        print(f"Error in tools/call: {str(e)}")
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": str(e)},
            "id": data.get("id", 1)
        }

@app.get("/player")
def get_player_stats(name: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Get player statcast data by name (optionally filter by date range: YYYY-MM-DD).
    """
    try:
        # Lazy load pybaseball only when this function is called
        pb = load_pybaseball()
        
        player_id = pb.playerid_lookup(name.split()[1], name.split()[0])
        if player_id.empty:
            raise ValueError(f"Player not found: {name}")
        
        player_id_mlbam = player_id.iloc[0]['key_mlbam']
        
        if start_date and end_date:
            stats = pb.statcast_batter(start_date, end_date, player_id_mlbam)
        else:
            # Default to last 30 days if no date range provided
            today = datetime.today().strftime('%Y-%m-%d')
            thirty_days_ago = (datetime.today() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            stats = pb.statcast_batter(thirty_days_ago, today, player_id_mlbam)
            
        if stats.empty:
            raise ValueError(f"No stats found for player: {name}")
            
        return {
            "player": name,
            "player_id": int(player_id_mlbam),
            "start_date": start_date,
            "end_date": end_date,
            "count": len(stats),
            "stats": stats.to_dict(orient="records")
        }
    except Exception as e:
        print(f"Error in get_player_stats: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/team_stats")
def get_team_stats(team: str, year: int, type: str = "batting"):
    """
    Get team stats for a given team and year. Type can be 'batting' or 'pitching'.
    """
    try:
        # Lazy load pybaseball only when this function is called
        pb = load_pybaseball()
        
        if type.lower() == "batting":
            stats = pb.team_batting(year)
        elif type.lower() == "pitching":
            stats = pb.team_pitching(year)
        else:
            raise ValueError("Invalid type. Use 'batting' or 'pitching'.")
            
        filtered = stats[stats['Team'].str.contains(team, case=False, na=False)]
        if filtered.empty:
            raise ValueError(f"No stats found for team: {team} in {year}.")
            
        return {
            "team": team,
            "year": year,
            "type": type,
            "count": len(filtered),
            "stats": filtered.to_dict(orient="records")
        }
    except Exception as e:
        print(f"Error in get_team_stats: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/leaderboard")
def get_leaderboard(stat: str, season: int, type: str = "batting"):
    """
    Get leaderboard for a given stat and season. Type can be 'batting' or 'pitching'.
    """
    try:
        # Lazy load pybaseball only when this function is called
        pb = load_pybaseball()
        
        if type.lower() == "batting":
            leaderboard = pb.batting_stats(season, season, qual=1, ind=0)
        elif type.lower() == "pitching":
            leaderboard = pb.pitching_stats(season, season, qual=1, ind=0)
        else:
            raise ValueError("Type must be 'batting' or 'pitching'.")
            
        if leaderboard.empty:
            raise ValueError(f"No leaderboard data found for {stat} in {season}.")
            
        # Filter for the requested stat column if present
        if stat not in leaderboard.columns:
            raise ValueError(f"Stat '{stat}' not found in leaderboard columns.")
            
        sorted_leaderboard = leaderboard.sort_values(by=stat, ascending=False).reset_index(drop=True)
        return {
            "stat": stat,
            "season": season,
            "type": type,
            "count": len(sorted_leaderboard),
            "leaderboard": sorted_leaderboard.to_dict(orient="records")
        }
    except Exception as e:
        print(f"Error in get_leaderboard: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
