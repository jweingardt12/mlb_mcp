from fastapi import FastAPI, Request, HTTPException, Query
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import importlib
import asyncio
import functools
import time
from fastapi.responses import JSONResponse

# Lazy imports - don't import pybaseballstats until needed
# This prevents slow startup times that can cause timeouts
pybaseball = None

# Configure FastAPI with minimal startup operations
app = FastAPI(
    title="MLB Stats MCP",
    description="MCP server exposing MLB/Fangraphs data via pybaseballstats.",
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
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            import pybaseballstats as pb
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
                "type": {"type": "string", "description": "Type of stats (batting or pitching), defaults to 'batting' if not provided.", "enum": ["batting", "pitching"]}
            },
            "required": ["team", "year"]
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
            print(f"[TOOLS_CALL_HANDLER] Received JRPC params: {params}") # Log incoming params
            # Handle gateway-style tool calls
            raw_tool_name = params.get("name")
            actual_tool_name = raw_tool_name
            if actual_tool_name and actual_tool_name.startswith("mlb-mcp-"):
                actual_tool_name = actual_tool_name.replace("mlb-mcp-", "", 1)

            # Try to get nested parameters, prioritizing 'parameters', then 'arguments'
            actual_tool_params = params.get("parameters")  # Standard MCP
            if actual_tool_params is None:
                actual_tool_params = params.get("arguments")  # Common alternative, as seen from client

            if actual_tool_params is None:
                # Fallback: if neither 'parameters' nor 'arguments' key exists,
                # assume parameters are at the top level of the 'params' object, excluding 'name'.
                actual_tool_params = {k: v for k, v in params.items() if k != "name"}
            
            # Ensure actual_tool_params is a dict, defaulting to empty if all attempts fail
            if not isinstance(actual_tool_params, dict):
                 actual_tool_params = {}

            print(f"[TOOLS_CALL_HANDLER] Derived tool_name: '{actual_tool_name}', derived_params: {actual_tool_params}") # Log derived params

            if actual_tool_name in ["get_player_stats", "get_team_stats", "get_leaderboard"]:
                result = call_tool(actual_tool_name, actual_tool_params)
                return {"jsonrpc": "2.0", "result": result, "id": rpc_id}
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Tool '{raw_tool_name}' (parsed as '{actual_tool_name}') not found or supported."},
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
        # Lazy load pybaseballstats only when this function is called
        pb = load_pybaseball()

        first, last = name.split()[0], " ".join(name.split()[1:])
        player_df = pb.retrosheet.player_lookup(first_name=first, last_name=last, return_pandas=True)
        if player_df.empty:
            raise ValueError(f"Player not found: {name}")

        player_id_mlbam = int(player_df.iloc[0]['key_mlbam'])

        if start_date and end_date:
            stats = pb.statcast.statcast_date_range_pitch_by_pitch(start_date, end_date, return_pandas=True)
        else:
            # Default to last 30 days if no date range provided
            today = datetime.today().strftime('%Y-%m-%d')
            thirty_days_ago = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
            stats = pb.statcast.statcast_date_range_pitch_by_pitch(thirty_days_ago, today, return_pandas=True)

        stats = stats[stats['batter'] == player_id_mlbam]
            
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
    current_server_year = datetime.now().year
    if year > current_server_year + 1: # Allow current year and next year only
        error_msg = f"Invalid year: {year}. Year cannot be more than 1 year in the future."
        print(error_msg)
        raise ValueError(error_msg)

    try:
        # Lazy load pybaseballstats only when this function is called
        pb = load_pybaseball()
        from pybaseballstats.bref_teams import BREFTeams
        from pybaseballstats.utils.fangraphs_consts import FangraphsTeams, FangraphsPitchingStatType
        from pybaseballstats import fangraphs

        def map_team(t: str):
            key = t.replace(" ", "_").upper()
            if key in BREFTeams.__members__:
                return BREFTeams[key], FangraphsTeams[key]
            for enum in BREFTeams:
                if enum.value.upper() == key:
                    fg = FangraphsTeams[enum.name]
                    return enum, fg
            raise ValueError(f"Invalid team: {t}")

        bref_team, fg_team = map_team(team)

        if type.lower() == "batting":
            stats = pb.bref_teams.team_standard_batting(bref_team, year, return_pandas=True)
        elif type.lower() == "pitching":
            stats = fangraphs.fangraphs_pitching_range(start_year=year, end_year=year,
                                                       stat_types=[FangraphsPitchingStatType.DASHBOARD],
                                                       team=fg_team, return_pandas=True)
        else:
            raise ValueError("Invalid type. Use 'batting' or 'pitching'.")

        if hasattr(stats, "to_pandas"):
            stats = stats.to_pandas()

        if stats.empty:
            raise ValueError(f"No stats found for team: {team} in {year}.")

        return {
            "team": team,
            "year": year,
            "type": type,
            "count": len(stats),
            "stats": stats.to_dict(orient="records")
        }
    except Exception as e:
        print(f"Error in get_team_stats: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/leaderboard")
def get_leaderboard(stat: str, season: int, type: str = "batting"):
    """
    Get leaderboard for a given stat and season. Type can be 'batting' or 'pitching'.
    """
    current_year = datetime.now().year
    if season > current_year + 1: # Allow current year and next year only
        error_msg = f"Invalid season: {season}. Year cannot be more than 1 year in the future."
        print(error_msg)
        raise ValueError(error_msg)

    # Mapping of friendly stat names to actual DataFrame columns
    STAT_COLUMN_MAP = {
        "avg_exit_velocity": "EV",  # Example, check actual column name in logs
        "exit_velocity_avg": "EV",
        "HR": "HR",
        "AVG": "AVG",
        "ERA": "ERA",
        # Add more mappings as needed
    }

    try:
        # Lazy load pybaseballstats only when this function is called
        pb = load_pybaseball()
        from pybaseballstats import fangraphs
        from pybaseballstats.utils.fangraphs_consts import (
            FangraphsBattingStatType,
            FangraphsPitchingStatType,
            FangraphsTeams,
        )

        if type.lower() == "batting":
            leaderboard = fangraphs.fangraphs_batting_range(
                start_year=season,
                end_year=season,
                stat_types=[FangraphsBattingStatType.DASHBOARD],
                team=FangraphsTeams.ALL,
                return_pandas=True,
            )
        elif type.lower() == "pitching":
            leaderboard = fangraphs.fangraphs_pitching_range(
                start_year=season,
                end_year=season,
                stat_types=[FangraphsPitchingStatType.DASHBOARD],
                team=FangraphsTeams.ALL,
                return_pandas=True,
            )
        else:
            raise ValueError("Type must be 'batting' or 'pitching'.")
            
        if hasattr(leaderboard, "to_pandas"):
            leaderboard = leaderboard.to_pandas()

        if leaderboard.empty:
            raise ValueError(f"No leaderboard data found for {stat} in {season}.")

        # Log available columns for debugging
        print(f"Available columns: {leaderboard.columns.tolist()}")

        # Map friendly stat name to actual column name
        column_name = STAT_COLUMN_MAP.get(stat, stat)
        if column_name not in leaderboard.columns:
            raise ValueError(f"Stat '{stat}' (mapped to '{column_name}') not found in leaderboard columns. Available columns: {leaderboard.columns.tolist()}")
            
        sorted_leaderboard = leaderboard.sort_values(by=column_name, ascending=False).reset_index(drop=True)
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

@app.get("/stats/options")
def get_stat_options(
    data_type: str = Query(..., description="Type of data: 'statcast', 'batting', 'pitching', 'team', 'division'"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    season: Optional[int] = None,
    player_name: Optional[str] = None,
    team: Optional[str] = None
):
    """
    Returns the available stat columns for a given data type and query.
    """
    pb = load_pybaseball()
    df = None

    if data_type == "statcast":
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="start_date and end_date required for statcast")
        df = pb.statcast(start_dt=start_date, end_dt=end_date, team=team)
    elif data_type == "batting":
        if not season:
            raise HTTPException(status_code=400, detail="season required for batting")
        df = pb.batting_stats(season)
    elif data_type == "pitching":
        if not season:
            raise HTTPException(status_code=400, detail="season required for pitching")
        df = pb.pitching_stats(season)
    elif data_type == "team":
        if not season:
            raise HTTPException(status_code=400, detail="season required for team stats")
        df = pb.team_batting(season)
    elif data_type == "division":
        if not season:
            raise HTTPException(status_code=400, detail="season required for division stats")
        df = pb.standings(season)
    else:
        raise HTTPException(status_code=400, detail="Unknown data_type")

    return {"columns": list(df.columns)}

@app.get("/statcast")
def get_statcast(start_date: str = Query(...), end_date: str = Query(...), team: Optional[str] = None):
    pb = load_pybaseball()
    df = pb.statcast(start_dt=start_date, end_dt=end_date, team=team)
    return df.to_dict(orient="records")

@app.get("/batting_stats")
def get_batting_stats(season: int = Query(...)):
    pb = load_pybaseball()
    df = pb.batting_stats(season)
    return df.to_dict(orient="records")

@app.get("/pitching_stats")
def get_pitching_stats(season: int = Query(...)):
    pb = load_pybaseball()
    df = pb.pitching_stats(season)
    return df.to_dict(orient="records")

@app.get("/standings")
def get_standings(season: int = Query(...)):
    pb = load_pybaseball()
    df = pb.standings(season)
    return df.to_dict(orient="records")

# --- BASEBALL LIBRARY ENDPOINTS ---

@app.get("/baseball/day")
def baseball_day(year: int, month: int, day: int, home: str = None, away: str = None):
    try:
        import baseball
        result = baseball.day(year, month, day, home=home, away=away)
        return JSONResponse(content={"games": [g.__dict__ for g in result]})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/games")
def baseball_games(years: int, months: str = None, days: str = None, home: str = None, away: str = None):
    try:
        import baseball
        months_list = [int(m) for m in months.split(",")] if months else None
        days_list = [int(d) for d in days.split(",")] if days else None
        result = baseball.games(years, months=months_list, days=days_list, home=home, away=away)
        # Flatten and serialize
        games = [g.__dict__ for day in result for g in day]
        return JSONResponse(content={"games": games})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/box_score")
def baseball_box_score(game_id: str):
    try:
        import baseball
        result = baseball.box_score(game_id)
        return JSONResponse(content={"box_score": result.__dict__})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/overview")
def baseball_overview(game_id: str):
    try:
        import baseball
        result = baseball.overview(game_id)
        return JSONResponse(content={"overview": result.__dict__})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/players")
def baseball_players(game_id: str):
    try:
        import baseball
        result = baseball.players(game_id)
        return JSONResponse(content={"players": result.__dict__})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/player_stats")
def baseball_player_stats(game_id: str):
    try:
        import baseball
        result = baseball.player_stats(game_id)
        return JSONResponse(content={"player_stats": result.__dict__})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/team_stats")
def baseball_team_stats(game_id: str):
    try:
        import baseball
        result = baseball.team_stats(game_id)
        return JSONResponse(content={"team_stats": result.__dict__})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/game_events")
def baseball_game_events(game_id: str):
    try:
        import baseball
        result = baseball.game_events(game_id)
        return JSONResponse(content={"game_events": [e.__dict__ for e in result]})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/important_dates")
def baseball_important_dates(year: int = None):
    try:
        import baseball
        result = baseball.important_dates(year)
        return JSONResponse(content={"important_dates": result.__dict__})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/league")
def baseball_league():
    try:
        import baseball
        result = baseball.league()
        return JSONResponse(content={"league": result.__dict__})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/teams")
def baseball_teams():
    try:
        import baseball
        result = baseball.teams()
        return JSONResponse(content={"teams": [t.__dict__ for t in result]})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/roster")
def baseball_roster(team_id: str):
    try:
        import baseball
        result = baseball.roster(team_id)
        return JSONResponse(content={"roster": result.__dict__})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/standings")
def baseball_standings():
    try:
        import baseball
        from datetime import datetime
        result = baseball.standings(datetime.now())
        return JSONResponse(content={"standings": result.__dict__})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/baseball/injury")
def baseball_injury():
    try:
        import baseball
        result = baseball.injury()
        return JSONResponse(content={"injury": result.__dict__})
    except ImportError:
        return JSONResponse(content={"error": "'baseball' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- BASEBALL-STATS-PYTHON ENDPOINTS ---

@app.get("/bsp/statcast_search")
def bsp_statcast_search(season: str, team: str = None, player_type: str = None, month: str = None):
    try:
        from baseball_stats_python import statcast_search
        result = statcast_search(season=season, team=team, player_type=player_type, month=month)
        return JSONResponse(content={"statcast_search": result.to_dict(orient="records")})
    except ImportError:
        return JSONResponse(content={"error": "'baseball-stats-python' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/bsp/minor_statcast_search")
def bsp_minor_statcast_search(season: str, level: str = None, month: str = None):
    try:
        from baseball_stats_python import minor_statcast_search
        result = minor_statcast_search(season=season, level=level, month=month)
        return JSONResponse(content={"minor_statcast_search": result.to_dict(orient="records")})
    except ImportError:
        return JSONResponse(content={"error": "'baseball-stats-python' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/bsp/mlbam_id_search")
def bsp_mlbam_id_search(name: str):
    try:
        from baseball_stats_python import mlbam_id_search
        result = mlbam_id_search(name)
        return JSONResponse(content={"mlbam_id_search": result.to_dict(orient="records")})
    except ImportError:
        return JSONResponse(content={"error": "'baseball-stats-python' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/bsp/statcast_batter_search")
def bsp_statcast_batter_search(batters_lookup: str, season: str = None, month: str = None):
    try:
        from baseball_stats_python import statcast_batter_search
        result = statcast_batter_search(batters_lookup=batters_lookup, season=season, month=month)
        return JSONResponse(content={"statcast_batter_search": result.to_dict(orient="records")})
    except ImportError:
        return JSONResponse(content={"error": "'baseball-stats-python' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/bsp/statcast_pitcher_search")
def bsp_statcast_pitcher_search(pitchers_lookup: str, season: str = None, month: str = None):
    try:
        from baseball_stats_python import statcast_pitcher_search
        result = statcast_pitcher_search(pitchers_lookup=pitchers_lookup, season=season, month=month)
        return JSONResponse(content={"statcast_pitcher_search": result.to_dict(orient="records")})
    except ImportError:
        return JSONResponse(content={"error": "'baseball-stats-python' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- MLB-StatsAPI VIDEO ENDPOINTS ---

@app.get("/mlb/highlights")
def mlb_game_highlights(game_id: int):
    try:
        import statsapi
        highlights = statsapi.game_highlights(game_id)
        return JSONResponse(content={"highlights": highlights})
    except ImportError:
        return JSONResponse(content={"error": "'MLB-StatsAPI' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/mlb/play_video")
def mlb_play_video(play_id: int):
    try:
        import statsapi
        # This is a placeholder; use the actual function to get play video data if available
        video_data = statsapi.get("video", {"playId": play_id})
        return JSONResponse(content={"video": video_data})
    except ImportError:
        return JSONResponse(content={"error": "'MLB-StatsAPI' package not installed."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
