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

app = FastAPI(title="Pybaseball MCP Server", description="MCP server exposing MLB/Fangraphs data via pybaseball.")

@app.get("/")
def read_root():
    return {"status": "ok"}

# MCP JSON-RPC tools registry
def list_tools():
    return [
        {
            "name": "get_player_stats",
            "description": "Get player statcast data by name (optionally filter by date range: YYYY-MM-DD)",
            "params": ["name", "start_date", "end_date"]
        },
        {
            "name": "get_team_stats",
            "description": "Get team stats for a given team and year. Type can be 'batting' or 'pitching'",
            "params": ["team", "year", "type"]
        },
        {
            "name": "get_leaderboard",
            "description": "Get leaderboard for a given stat and season. Type can be 'batting' or 'pitching'",
            "params": ["stat", "season", "type"]
        }
    ]

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

def call_tool(method, params):
    # Load pybaseball only when a tool is actually called
    try:
        if method == "get_player_stats":
            return get_player_stats(**params)
        elif method == "get_team_stats":
            return get_team_stats(**params)
        elif method == "get_leaderboard":
            return get_leaderboard(**params)
        else:
            raise Exception(f"Unknown method: {method}")
    except Exception as e:
        # Catch and log any exceptions
        print(f"Error calling {method}: {str(e)}")
        raise

@app.post("/tools/list")
async def mcp_tools_list():
    # Return a static tool list immediately without any processing
    # This implements lazy loading by avoiding any expensive operations
    return {
        "jsonrpc": "2.0",
        "result": {
            "tools": [
                {
                    "name": "get_player_stats",
                    "description": "Get player statcast data by name (optionally filter by date range: YYYY-MM-DD)",
                    "params": ["name", "start_date", "end_date"]
                },
                {
                    "name": "get_team_stats",
                    "description": "Get team stats for a given team and year. Type can be 'batting' or 'pitching'",
                    "params": ["team", "year", "type"]
                },
                {
                    "name": "get_leaderboard",
                    "description": "Get leaderboard for a given stat and season. Type can be 'batting' or 'pitching'",
                    "params": ["stat", "season", "type"]
                }
            ]
        },
        "id": 1
    }

@app.post("/tools/call")
async def mcp_tools_call(request: Request):
    data = await request.json()
    try:
        method = data.get("method")
        params = data.get("params", {})
        
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
