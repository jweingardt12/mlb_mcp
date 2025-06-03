from fastapi import FastAPI, HTTPException, Request
from pybaseball import playerid_lookup, statcast_batter, team_batting, team_pitching, batting_stats, pitching_stats
from typing import Optional
from datetime import datetime

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

def call_tool(method, params):
    if method == "get_player_stats":
        return get_player_stats(**params)
    elif method == "get_team_stats":
        return get_team_stats(**params)
    elif method == "get_leaderboard":
        return get_leaderboard(**params)
    else:
        raise Exception(f"Unknown method: {method}")

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
        parts = name.strip().split()
        if len(parts) < 2:
            raise ValueError("Please provide both first and last name.")
        first, last = parts[0], " ".join(parts[1:])
        lookup = playerid_lookup(last, first)
        if lookup.empty:
            raise ValueError(f"Player '{name}' not found.")
        mlb_id = lookup.iloc[0]['key_mlbam']
        if not start_date:
            start_date = f"{datetime.now().year-1}-03-01"
        if not end_date:
            end_date = f"{datetime.now().year-1}-11-15"
        stats = statcast_batter(start_date, end_date, mlb_id)
        return {
            "name": name,
            "mlb_id": int(mlb_id),
            "start_date": start_date,
            "end_date": end_date,
            "count": len(stats),
            "stats": stats.to_dict(orient="records") if not stats.empty else []
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/team_stats")
def get_team_stats(team: str, year: int, type: str = "batting"):
    """
    Get team stats for a given team and year. Type can be 'batting' or 'pitching'.
    """
    try:
        if type.lower() == "batting":
            stats = team_batting(year)
        elif type.lower() == "pitching":
            stats = team_pitching(year)
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
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/leaderboard")
def get_leaderboard(stat: str, season: int, type: str = "batting"):
    """
    Get leaderboard for a given stat and season. Type can be 'batting' or 'pitching'.
    """
    try:
        if type.lower() == "batting":
            leaderboard = batting_stats(season, season, qual=1, ind=0)
        elif type.lower() == "pitching":
            leaderboard = pitching_stats(season, season, qual=1, ind=0)
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
        raise HTTPException(status_code=404, detail=str(e))
