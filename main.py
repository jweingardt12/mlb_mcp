from fastapi import FastAPI, Request, HTTPException, Query, Body
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import importlib
import asyncio
import functools
import time
from fastapi.responses import JSONResponse
import pandas as pd
import requests
import numpy as np
import traceback

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
    if pybaseball is not None:
        return pybaseball
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            import pybaseballstats as pb
        pybaseball = pb
        pybaseball._source = "pybaseballstats"
    except ImportError:
        try:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                import pybaseball as pb
            pybaseball = pb
            pybaseball._source = "pybaseball"
        except ImportError:
            pybaseball = None
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
    pb = load_pybaseball() # Assuming load_pybaseball() is robust or its errors can propagate

    try:
        if tool_name == "get_player_stats":
            return get_player_stats(**params)
        elif tool_name == "get_team_stats":
            return get_team_stats(**params)
        elif tool_name == "get_leaderboard":
            return get_leaderboard(**params)
        elif tool_name == "mlb_video_search":
            # Call the endpoint logic directly
            # params: query, limit, page
            from fastapi import Request
            class DummyRequest:
                def __init__(self, json_data):
                    self._json = json_data
                async def json(self):
                    return self._json
            # Call the async endpoint in a sync context
            import asyncio
            coro = mlb_video_search(DummyRequest(params))
            if asyncio.iscoroutine(coro):
                return asyncio.get_event_loop().run_until_complete(coro)
            else:
                return coro
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    except Exception as e:
        error_message = f"Error executing tool '{tool_name}' with params '{params}': {str(e)}"
        print(error_message)
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
        "description": "Get leaderboard for a given stat and season. Use this for stats/leaderboards, not for video highlights.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "stat": {"type": "string", "description": "Statistic to get leaderboard for (e.g., 'HR', 'AVG', 'ERA')"},
                "season": {"type": "integer", "description": "Season year to get leaderboard for"},
                "type": {"type": "string", "description": "Type of leaderboard (batting or pitching)", "enum": ["batting", "pitching"]},
                "limit": {"type": "integer", "description": "Number of results to return", "default": 10},
                "as_text": {"type": "boolean", "description": "Return a text summary of the leaderboard", "default": False},
                "month": {"type": "integer", "description": "Month to filter leaderboard", "enum": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "default": None},
                "day": {"type": "integer", "description": "Day to filter leaderboard", "enum": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], "default": None},
                "date": {"type": "string", "description": "Date to filter leaderboard", "format": "YYYY-MM-DD", "default": None},
                "week": {"type": "integer", "description": "Week to filter leaderboard", "enum": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], "default": None},
                "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format", "format": "YYYY-MM-DD", "default": None},
                "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format", "format": "YYYY-MM-DD", "default": None},
                "result": {"type": "string", "description": "Filter by result type, e.g., 'Home Run' (optional)"},
                "order": {"type": "string", "description": "Sort order: 'asc' or 'desc' (optional, default desc)"}
            },
            "required": ["stat", "season"]
        }
    },
    {
        "name": "mlb_video_search",
        "description": "Search MLB video highlights using the MLB Film Room API. Use this ONLY for video/highlight queries, not for stats or leaderboards.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "MLB Film Room search query (e.g., 'Season = [2023] AND HitResult = [\"Home Run\"]')"},
                "limit": {"type": "integer", "description": "Number of results to return", "default": 10},
                "page": {"type": "integer", "description": "Page number for pagination", "default": 0}
            },
            "required": ["query"]
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

            if actual_tool_name in ["get_player_stats", "get_team_stats", "get_leaderboard", "mlb_video_search"]:
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
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Player name must be provided.")
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

# Add a mapping for natural language stat queries
NATURAL_LANGUAGE_STAT_MAP = {
    # Exit velocity
    "hardest hit balls": "EV",
    "hardest hit ball": "EV",
    "hardest hit": "EV",
    "highest exit velocity": "EV",
    "top exit velocity": "EV",
    "top exit velo": "EV",
    "exit velo": "EV",
    "exit velocity": "EV",
    "max exit velocity": "maxEV",
    "maximum exit velocity": "maxEV",
    "maxev": "maxEV",
    "hardest hit home run": "EV",
    # Add more as needed
}

@app.get("/leaderboard")
def get_leaderboard(
    stat: str,
    season: int,
    type: str = "batting",
    limit: int = Query(10, description="Number of results to return"),
    as_text: bool = False,
    month: int = None,
    day: int = None,
    date: str = None,
    week: int = None,
    start_date: str = None,
    end_date: str = None,
    result: str = Query(None, description="Filter by result type, e.g., 'Home Run' (optional)"),
    order: str = Query("desc", description="Sort order: 'asc' or 'desc' (optional, default desc)")
):
    """
    Get leaderboard for a given stat and season. Type can be 'batting' or 'pitching'.
    Optional filters:
      - month (1-12)
      - day (1-31)
      - date (YYYY-MM-DD)
      - week (1-53)
      - start_date, end_date (YYYY-MM-DD)
      - result (e.g., 'Home Run')
      - order ('asc' or 'desc')
    If possible, includes a video resource for each leaderboard row.
    """
    current_year = datetime.now().year
    if season > current_year + 1: # Allow current year and next year only
        error_msg = f"Invalid season: {season}. Year cannot be more than 1 year in the future."
        print(error_msg)
        return {"content": [], "error": error_msg}

    # Normalize stat using natural language mapping first
    stat_lower = stat.lower().strip()
    if stat_lower in NATURAL_LANGUAGE_STAT_MAP:
        stat = NATURAL_LANGUAGE_STAT_MAP[stat_lower]
    # Statcast metrics that should redirect to statcast_leaderboard
    STATCAST_METRICS = {"ev", "exitevelocity", "exit_velocity", "exitvelocity", "maxev", "max_exit_velocity", "launch_speed", "launchangle", "launch_angle", "hit_distance", "hitdistance", "distance"}
    stat_norm = stat.lower().replace("_", "")
    # Counting stats that can be aggregated from Statcast
    COUNTING_STATS = {"hr", "home_run", "home runs", "home run", "homeruns", "homers"}
    # If user requests a counting stat (like HR) for a specific month, day, or date range, use Statcast
    if (stat_norm in COUNTING_STATS or stat_norm == "hr") and (month or day or date or (start_date and end_date)):
        try:
            # Determine date range
            if date:
                start = end = date
            elif start_date and end_date:
                start, end = start_date, end_date
            else:
                # If only month is specified, use the whole month
                if month:
                    start = f"{season}-{month:02d}-01"
                    # Get last day of month
                    if month == 12:
                        end = f"{season}-12-31"
                    else:
                        end = (datetime(season, month+1, 1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    # Fallback to season
                    start = f"{season}-03-01"
                    end = f"{season}-11-30"
            # Query Statcast for the date range
            pb = load_pybaseball()
            df = pb.statcast(start_dt=start, end_dt=end)
            if df.empty:
                return {"content": []}
            # Only keep batted balls
            df = df[df["events"].notnull()]
            # Filter for home runs
            df_hr = df[df["events"].str.lower().str.contains("home_run")]
            # Group by batter, count HRs
            hr_counts = df_hr.groupby("batter").size().reset_index(name="HR")
            # Get player names and teams
            from pybaseball.playerid_lookup import playerid_reverse_lookup
            def get_player_name(pid):
                lookup = playerid_reverse_lookup([pid], key_type="mlbam")
                if not lookup.empty:
                    return f'{lookup.iloc[0]["name_first"]} {lookup.iloc[0]["name_last"]}'
                return str(pid)
            # Get team for each batter (use the most common team in the HRs for that period)
            def get_team(pid):
                # Infer team from inning_topbot, home_team, away_team
                batter_rows = df_hr[df_hr["batter"] == pid]
                teams = []
                for _, row in batter_rows.iterrows():
                    if "inning_topbot" in row and "home_team" in row and "away_team" in row:
                        if row["inning_topbot"] == "Top":
                            teams.append(row["away_team"])
                        elif row["inning_topbot"] == "Bot":
                            teams.append(row["home_team"])
                if teams:
                    # Return the most common team
                    from collections import Counter
                    return Counter(teams).most_common(1)[0][0]
                return None
            hr_counts["player_name"] = hr_counts["batter"].apply(get_player_name)
            hr_counts["team"] = hr_counts["batter"].apply(get_team)
            hr_counts = hr_counts.sort_values(by="HR", ascending=False).head(limit)
            records = hr_counts[["player_name", "team", "HR"]].to_dict(orient="records")
            if len(records) > 3:
                return {"content": records}
            else:
                return {"content": records}
        except Exception as e:
            print(f"Error in HR aggregation: {e}")
            traceback.print_exc()
            return {"content": [], "error": str(e)}

    # Friendly mapping for common aliases
    FRIENDLY_STAT_MAP = {
        # Batting
        "exit_velocity": "EV", "exitvelocity": "EV", "exit_velocity_avg": "EV", "avg_exit_velocity": "EV", "ev": "EV",
        "launch_angle": "LA", "avg_launch_angle": "LA", "la": "LA",
        "barrels": "Barrels", "barrel_count": "Barrels",
        "barrel_rate": "Barrel%", "barrel%": "Barrel%",
        "max_exit_velocity": "maxEV", "maxev": "maxEV", "hardest_hit": "maxEV",
        "hard_hit": "HardHit", "hardhit": "HardHit", "hard_hit_count": "HardHit",
        "hard_hit_rate": "HardHit%", "hardhit%": "HardHit%", "hard_hit%": "HardHit%",
        "events": "Events",
        "xba": "xBA", "expected_ba": "xBA",
        "xslg": "xSLG", "expected_slg": "xSLG",
        "xwoba": "xwOBA", "expected_woba": "xwOBA",
        "woba": "wOBA",
        "wrc+": "wRC+", "wrcplus": "wRC+",
        "war": "WAR",
        "hr": "HR", "home_runs": "HR",
        "avg": "AVG", "batting_average": "AVG",
        "obp": "OBP",
        "slg": "SLG",
        "iso": "ISO",
        "babip": "BABIP",
        "sb": "SB", "stolen_bases": "SB",
        "rbi": "RBI",
        "r": "R",
        "ab": "AB",
        "pa": "PA",
        "so": "SO", "k": "SO", "strikeouts": "SO",
        "bb": "BB", "walks": "BB",
        "age": "Age",
        "team": "Team",
        "name": "Name", "player": "Name",
        # Pitching
        "era": "ERA",
        "wins": "W", "w": "W",
        "losses": "L", "l": "L",
        "gs": "GS",
        "cg": "CG",
        "sho": "ShO",
        "sv": "SV",
        "ip": "IP",
        "tbf": "TBF",
        "er": "ER",
        "whip": "WHIP",
        "fip": "FIP",
        "xfip": "xFIP",
        "k9": "K/9",
        "bb9": "BB/9",
        "kbb": "K/BB",
        "h9": "H/9",
        "hr9": "HR/9",
        "lob%": "LOB%",
        "siera": "SIERA",
        # Add more as needed
    }

    def normalize_key(key):
        return key.lower().replace("_", "")

    # Result code mapping (Fangraphs/pybaseball common codes)
    RESULT_CODE_MAP = {
        0: "Unknown",
        1: "Out",
        2: "Single",
        3: "Double",
        4: "Triple",
        5: "Error",
        6: "Fielder's Choice",
        7: "Walk",
        8: "Intentional Walk",
        9: "Hit By Pitch",
        10: "Interference",
        11: "Sacrifice Hit",
        12: "Sacrifice Fly",
        13: "Catcher Interference",
        14: "Field Error",
        15: "Bunt",
        16: "Groundout",
        17: "Flyout",
        18: "Lineout",
        19: "Popout",
        20: "Strikeout",
        21: "Double Play",
        22: "Triple Play",
        23: "Home Run",
        # Add more as needed
    }

    # --- Helper function for filtering leaderboard DataFrames ---
    def apply_leaderboard_filters(df, month=None, day=None, week=None, date=None, start_date=None, end_date=None, result=None):
        # Apply time filters if columns exist
        if month is not None and "Month" in df.columns:
            df = df[df["Month"] == month]
        if day is not None and "Day" in df.columns:
            df = df[df["Day"] == day]
        if week is not None and "Week" in df.columns:
            df = df[df["Week"] == week]
        if date is not None and "Date" in df.columns:
            df = df[df["Date"] == date]
        if start_date is not None and end_date is not None and "Date" in df.columns:
            df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        # Apply result filter if present
        if result is not None:
            # Try to match in common result columns
            result_cols = [col for col in ["Result", "HitResult", "HR", "events"] if col in df.columns]
            if result_cols:
                col = result_cols[0]
                # Case-insensitive match for string columns
                if df[col].dtype == object:
                    df = df[df[col].astype(str).str.lower() == result.lower()]
                else:
                    # Try to map result string to code if possible
                    code = None
                    for k, v in RESULT_CODE_MAP.items():
                        if v.lower() == result.lower():
                            code = k
                            break
                    if code is not None:
                        df = df[df[col] == code]
        return df

    try:
        pb = load_pybaseball()
        if pb is None:
            return {"content": [], "error": "Neither pybaseballstats nor pybaseball is installed."}
        # Use pybaseballstats if loaded
        if getattr(pb, "_source", None) == "pybaseballstats":
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
                return {"content": [], "error": "Type must be 'batting' or 'pitching'."}
            if hasattr(leaderboard, "to_pandas"):
                leaderboard = leaderboard.to_pandas()
            if leaderboard.empty:
                return {"content": []}
            print(f"Available columns: {leaderboard.columns.tolist()}")
            # Build dynamic mapping for all columns
            norm_col_map = {normalize_key(col): col for col in leaderboard.columns}
            stat_norm = normalize_key(stat)
            # Prefer friendly mapping, else any available column
            column_name = FRIENDLY_STAT_MAP.get(stat_norm, None)
            if not column_name:
                column_name = norm_col_map.get(stat_norm, stat)
            if column_name not in leaderboard.columns:
                return {"content": [], "error": f"Stat '{stat}' (mapped to '{column_name}') not found. Available columns: {leaderboard.columns.tolist()}"}
            # --- Apply all filters before sorting ---
            leaderboard = apply_leaderboard_filters(
                leaderboard, month=month, day=day, week=week, date=date, start_date=start_date, end_date=end_date, result=result
            )
            # --- Sort by requested stat and order ---
            ascending = order == "asc"
            sorted_leaderboard = leaderboard.sort_values(by=column_name, ascending=ascending).reset_index(drop=True)
            name_columns = [col for col in sorted_leaderboard.columns if col.lower() in ["name", "player", "player_name"]]
            if name_columns:
                sorted_leaderboard = sorted_leaderboard.rename(columns={name_columns[0]: "Name"})
            else:
                sorted_leaderboard["Name"] = None
            sorted_leaderboard = sorted_leaderboard.replace([np.inf, -np.inf, float('inf'), float('-inf')], np.nan)
            sorted_leaderboard = sorted_leaderboard.where(pd.notnull(sorted_leaderboard), None)
            sorted_leaderboard = sorted_leaderboard.applymap(safe_json)
            records = sorted_leaderboard.to_dict(orient="records")
            if as_text:
                lines = [f"{season} {type.title()} Leaderboard for {stat} (column: {column_name}):"]
                for i, row in enumerate(records[:limit]):
                    name = row.get("Name", "N/A")
                    team = row.get("Team", "N/A")
                    value = row.get(column_name, "N/A")
                    lines.append(f"{i+1}. Name: {name}, Team: {team}, {column_name}: {value}")
                return {"content": [{"type": "text", "text": "\n".join(lines)}]}
            # Attach video to each record if possible
            for row in records:
                row["video"] = find_video_for_row(row, date)
            # Only keep primary fields for output
            def extract_primary_fields(row):
                result = {}
                # Player name
                result["player_name"] = row.get("Name")
                # Pitcher (if available)
                pitcher = row.get("Pitcher") or row.get("OpposingPitcher") or row.get("PitcherName")
                if pitcher:
                    result["pitcher"] = pitcher
                # Exit velocity (EV or maxEV)
                ev = row.get("EV") or row.get("maxEV")
                if ev:
                    result["exit_velocity"] = ev
                # Hit distance (if available)
                hit_distance = row.get("HitDistance") or row.get("Distance")
                if hit_distance:
                    result["hit_distance"] = hit_distance
                # Result (HR, 2B, etc.)
                result_col = row.get("Result") or row.get("HitResult") or row.get("HR")
                # Map result code to string if possible
                if isinstance(result_col, int) and result_col in RESULT_CODE_MAP:
                    result["result"] = RESULT_CODE_MAP[result_col]
                elif result_col:
                    result["result"] = str(result_col)
                # Opponent
                opponent = row.get("Opponent") or row.get("OpposingTeam") or row.get("Opp")
                if opponent:
                    result["opponent"] = opponent
                # Date
                date_val = row.get("Date")
                if date_val:
                    result["date"] = date_val
                # Add other relevant play-level fields if available
                for key in ["Event", "PlayDesc", "PlayDescription", "Inning", "Team"]:
                    if row.get(key) is not None:
                        result[key.lower()] = row.get(key)
                # Video
                result["video"] = row.get("video")
                return result
            primary_records = [extract_primary_fields(row) for row in records[:limit]]
            if len(primary_records) > 3:
                return {"content": primary_records}
            else:
                return {"content": primary_records}
        # Use pybaseball if loaded
        elif getattr(pb, "_source", None) == "pybaseball":
            if type.lower() == "batting":
                df = pb.batting_stats(season)
            elif type.lower() == "pitching":
                df = pb.pitching_stats(season)
            else:
                return {"content": [], "error": "Type must be 'batting' or 'pitching'."}
            if df.empty:
                return {"content": []}
            print(f"Available columns: {df.columns.tolist()}")
            norm_col_map = {normalize_key(col): col for col in df.columns}
            stat_norm = normalize_key(stat)
            column_name = FRIENDLY_STAT_MAP.get(stat_norm, None)
            if not column_name:
                column_name = norm_col_map.get(stat_norm, stat)
            if column_name not in df.columns:
                return {"content": [], "error": f"Stat '{stat}' (mapped to '{column_name}') not found. Available columns: {df.columns.tolist()}`"}
            # --- Apply all filters before sorting ---
            df = apply_leaderboard_filters(
                df, month=month, day=day, week=week, date=date, start_date=start_date, end_date=end_date, result=result
            )
            # --- Sort by requested stat and order ---
            ascending = order == "asc"
            sorted_df = df.sort_values(by=column_name, ascending=ascending).reset_index(drop=True)
            name_columns = [col for col in sorted_df.columns if col.lower() in ["name", "player", "player_name"]]
            if name_columns:
                sorted_df = sorted_df.rename(columns={name_columns[0]: "Name"})
            else:
                sorted_df["Name"] = None
            sorted_df = sorted_df.replace([np.inf, -np.inf, float('inf'), float('-inf')], np.nan)
            sorted_df = sorted_df.where(pd.notnull(sorted_df), None)
            sorted_df = sorted_df.applymap(safe_json)
            records = sorted_df.to_dict(orient="records")
            if as_text:
                lines = [f"{season} {type.title()} Leaderboard for {stat} (column: {column_name}):"]
                for i, row in enumerate(records[:limit]):
                    name = row.get("Name", "N/A")
                    team = row.get("Team", "N/A")
                    value = row.get(column_name, "N/A")
                    lines.append(f"{i+1}. Name: {name}, Team: {team}, {column_name}: {value}")
                return {"content": [{"type": "text", "text": "\n".join(lines)}]}
            # Attach video to each record if possible
            for row in records:
                row["video"] = find_video_for_row(row, date)
            # Only keep primary fields for output
            def extract_primary_fields(row):
                result = {}
                # Player name
                result["player_name"] = row.get("Name")
                # Pitcher (if available)
                pitcher = row.get("Pitcher") or row.get("OpposingPitcher") or row.get("PitcherName")
                if pitcher:
                    result["pitcher"] = pitcher
                # Exit velocity (EV or maxEV)
                ev = row.get("EV") or row.get("maxEV")
                if ev:
                    result["exit_velocity"] = ev
                # Hit distance (if available)
                hit_distance = row.get("HitDistance") or row.get("Distance")
                if hit_distance:
                    result["hit_distance"] = hit_distance
                # Result (HR, 2B, etc.)
                result_col = row.get("Result") or row.get("HitResult") or row.get("HR")
                # Map result code to string if possible
                if isinstance(result_col, int) and result_col in RESULT_CODE_MAP:
                    result["result"] = RESULT_CODE_MAP[result_col]
                elif result_col:
                    result["result"] = str(result_col)
                # Opponent
                opponent = row.get("Opponent") or row.get("OpposingTeam") or row.get("Opp")
                if opponent:
                    result["opponent"] = opponent
                # Date
                date_val = row.get("Date")
                if date_val:
                    result["date"] = date_val
                # Add other relevant play-level fields if available
                for key in ["Event", "PlayDesc", "PlayDescription", "Inning", "Team"]:
                    if row.get(key) is not None:
                        result[key.lower()] = row.get(key)
                # Video
                result["video"] = row.get("video")
                return result
            primary_records = [extract_primary_fields(row) for row in records[:limit]]
            if len(primary_records) > 3:
                return {"content": primary_records}
            else:
                return {"content": primary_records}
        else:
            return {"content": [], "error": "Unknown baseball package loaded."}
    except Exception as e:
        print(f"Error in get_leaderboard: {str(e)}")
        return {"content": [], "error": str(e)}

def find_video_for_row(row, default_date=None):
    import requests
    player = row.get("Name")
    row_date = row.get("Date") or default_date
    result_val = row.get("Result") or row.get("HitResult")
    hit_distance = row.get("HitDistance") or row.get("Distance")
    attempts = []

    # Most strict: all fields, wide hit distance
    if player and row_date and result_val and hit_distance:
        attempts.append(
            f'Player = ["{player}"] AND Date = ["{row_date}"] AND HitResult = ["{result_val}"] AND HitDistance >= {hit_distance - 10} AND HitDistance <= {hit_distance + 10} AND video'
        )
    # Player + Date + video
    if player and row_date:
        attempts.append(
            f'Player = ["{player}"] AND Date = ["{row_date}"] AND video'
        )
    # Player + video
    if player:
        attempts.append(
            f'Player = ["{player}"] AND video'
        )

    for video_query in attempts:
        try:
            resp = requests.post(
                "http://localhost:8000/mlb/video_search",
                json={"query": video_query, "limit": 1}
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("content", [])
                if content and isinstance(content, list):
                    return content[0]
        except Exception as e:
            print(f"Error finding video for {player} on {row_date}: {e}")
    return None

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

@app.post("/mlb/video_search")
async def mlb_video_search(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        limit = data.get("limit", 10)
        page = data.get("page", 0)
        # Guard clause for video-related queries
        video_keywords = ["video", "highlight", "clip", "watch", "playback", "film room"]
        if not any(kw in query.lower() for kw in video_keywords):
            return {
                "content": [],
                "error": "This tool is for video/highlight queries only. For stats or leaderboards, use the get_leaderboard tool."
            }
        url = "https://fastball-gateway.mlb.com/graphql"
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://www.mlb.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://www.mlb.com/video/",
            "sec-ch-ua": '"Chromium";v="137", "Not/A)Brand";v="24"',
            "sec-ch-ua-mobile": "?1",
            "sec-ch-ua-platform": '"Android"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "sec-gpc": "1",
            "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36"
        }
        body = {
            "query": "query Search($query: String!, $page: Int, $limit: Int, $feedPreference: FeedPreference, $languagePreference: LanguagePreference, $contentPreference: ContentPreference, $forgeInstance: ForgeType = MLB, $queryType: QueryType = STRUCTURED) { search(query: $query, limit: $limit, page: $page, feedPreference: $feedPreference, languagePreference: $languagePreference, contentPreference: $contentPreference, forgeInstance: $forgeInstance, queryType: $queryType) { plays { mediaPlayback { id slug blurb date description title canAddToReel feeds { type duration image { altText templateUrl cuts { width src __typename } __typename } playbacks { name __typename } __typename } playInfo { balls strikes outs inning inningHalf pitchSpeed pitchType exitVelocity hitDistance launchAngle spinRate scoreDifferential gamePk runners { first second third __typename } teams { away { name shortName triCode __typename } home { name shortName triCode __typename } batting { name shortName triCode __typename } pitching { name shortName triCode __typename } __typename } players { pitcher { id name lastName playerHand __typename } batter { id name lastName playerHand __typename } __typename } __typename } keywordsDisplay { slug displayName __typename } __typename } __typename } total __typename } }",
            "operationName": "Search",
            "variables": {
                "forgeInstance": "MLB",
                "queryType": None,
                "query": query,
                "limit": limit,
                "page": page,
                "languagePreference": "EN",
                "contentPreference": "MIXED"
            }
        }
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        raw = response.json()
        # Flatten the response
        videos = []
        plays = raw.get("data", {}).get("search", {}).get("plays", [])
        for play in plays:
            for mp in play.get("mediaPlayback", []):
                # Get main feed (CMS preferred)
                feeds = mp.get("feeds", [])
                main_feed = next((f for f in feeds if f.get("type") == "CMS"), feeds[0] if feeds else None)
                # Get video URLs (playbacks)
                video_urls = []
                if main_feed and "playbacks" in main_feed:
                    video_urls = [pb.get("name") for pb in main_feed["playbacks"] if pb.get("name")]
                # Get image URL (pick largest cut if available)
                image_url = None
                if main_feed and "image" in main_feed:
                    cuts = main_feed["image"].get("cuts", [])
                    if cuts:
                        # Pick the largest width
                        largest = max(cuts, key=lambda c: c.get("width", 0))
                        image_url = largest.get("src")
                    else:
                        image_url = main_feed["image"].get("templateUrl")
                # Statcast info
                play_info = mp.get("playInfo", {})
                statcast = {
                    k: play_info.get(k)
                    for k in ["exitVelocity", "hitDistance", "launchAngle", "pitchSpeed", "pitchType", "inning", "inningHalf", "balls", "strikes", "outs"]
                    if k in play_info
                }
                # Player and team info
                player_name = None
                team = None
                if "players" in play_info and "batter" in play_info["players"]:
                    player_name = play_info["players"]["batter"].get("name")
                if "teams" in play_info and "batting" in play_info["teams"]:
                    team = play_info["teams"]["batting"].get("name")
                slug = mp.get("slug")
                uri = f"https://www.mlb.com/video/{slug}" if slug else None
                text = mp.get("title") or mp.get("description") or mp.get("blurb")
                video_obj = {
                    "id": mp.get("id"),
                    "slug": slug,
                    "title": mp.get("title"),
                    "description": mp.get("description"),
                    "date": mp.get("date"),
                    "blurb": mp.get("blurb"),
                    "player_name": player_name,
                    "team": team,
                    "video_urls": video_urls,
                    "image_url": image_url,
                    "statcast": statcast,
                    "uri": uri,
                    "text": text
                }
                videos.append({"type": "resource", "resource": video_obj})
        if len(videos) > 3:
            return {"content": videos}
        else:
            return {"content": videos}
    except Exception as e:
        return {"error": str(e)}

def safe_json(val):
    try:
        if isinstance(val, float):
            if not np.isfinite(val):
                return None
        return val
    except Exception:
        return None

def sanitize_json(obj):
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    else:
        return obj

# Example MLB video search query for hardest hit balls on 4-seam fastballs in 2023
# GET /mlb/video_search?queries=PitchType = ["FF"] AND Season = ["2023"] Order By ExitVelocity DESC

# --- Statcast leaderboard logic as a plain function ---
def statcast_leaderboard(
    start_date: str,
    end_date: str,
    limit: int = 10,
    min_ev: float = None,
    result: str = None
):
    pb = load_pybaseball()
    if pb is None:
        return {"content": [], "error": "Neither pybaseballstats nor pybaseball is installed."}
    try:
        df = pb.statcast(start_dt=start_date, end_dt=end_date)
        if df.empty:
            return {"content": []}
        df = df[df["launch_speed"].notnull()]
        if min_ev is not None:
            df = df[df["launch_speed"] >= min_ev]
        if result is not None:
            mask = df["events"].str.lower().str.contains(result.lower(), na=False) | df["description"].str.lower().str.contains(result.lower(), na=False)
            df = df[mask]
        df = df.sort_values(by="launch_speed", ascending=False).reset_index(drop=True)
        records = []
        batter_id_to_name = {}
        def get_player_name(player_id):
            from pybaseball.playerid_lookup import playerid_reverse_lookup
            try:
                player_id_int = int(player_id)
            except Exception:
                player_id_int = player_id
            if player_id_int in batter_id_to_name:
                return batter_id_to_name[player_id_int]
            try:
                lookup = playerid_reverse_lookup([player_id_int], key_type='mlbam')
                if not lookup.empty:
                    name = f"{lookup.iloc[0]['name_first']} {lookup.iloc[0]['name_last']}"
                    batter_id_to_name[player_id_int] = name
                    return name
            except Exception as e:
                print(f"Error reverse looking up player name for ID {player_id_int}: {e}")
            batter_id_to_name[player_id_int] = str(player_id_int)
            return str(player_id_int)
        PITCH_TYPE_MAP = {
            'FF': 'Four-Seam Fastball', 'FT': 'Two-Seam Fastball', 'SI': 'Sinker', 'FC': 'Cutter',
            'FS': 'Splitter', 'FO': 'Forkball', 'SL': 'Slider', 'CH': 'Changeup', 'CU': 'Curveball',
            'KC': 'Knuckle Curve', 'KN': 'Knuckleball', 'EP': 'Eephus', 'SC': 'Screwball', 'ST': 'Sweeper',
            'SV': 'Slurve', 'CS': 'Slow Curve', 'UN': 'Unknown',
        }
        for _, row in df.head(limit).iterrows():
            batter_id = row.get("batter")
            pitcher_id = row.get("pitcher")
            home_team = row.get("home_team")
            away_team = row.get("away_team")
            inning_topbot = row.get("inning_topbot")
            if inning_topbot == "Top":
                batter_team = away_team
                opponent_team = home_team
            else:
                batter_team = home_team
                opponent_team = away_team
            rec = {
                "batter_name": get_player_name(batter_id),
                "pitcher_name": get_player_name(pitcher_id),
                "batter_team": batter_team,
                "opponent_team": opponent_team,
                "date": row.get("game_date"),
                "exit_velocity": row.get("launch_speed"),
                "hit_distance": row.get("hit_distance_sc"),
                "result": row.get("events") or row.get("description"),
                "inning": row.get("inning"),
                "pitch_type": row.get("pitch_type"),
                "pitch_type_name": PITCH_TYPE_MAP.get(row.get("pitch_type"), row.get("pitch_type")),
                "pitch_velocity": row.get("release_speed"),
            }
            mlb_video_url = None
            if rec["result"] and "home_run" in str(rec["result"]).lower() and rec["batter_name"] and rec["date"]:
                batter_slug = rec["batter_name"].replace(" ", "-").lower()
                date_str = str(rec["date"]).split("T")[0]
                search_query = f"{rec['batter_name']} home run {date_str}"
                mlb_video_url = f"https://www.mlb.com/video/search?q={search_query.replace(' ', '%20')}"
            video_query_parts = [
                f'Player = ["{rec["batter_name"]}"]',
                f'Date = ["{rec["date"]}"]',
                'video'
            ]
            if rec["result"]:
                video_query_parts.append(f'HitResult = ["{rec["result"]}"]')
            if rec["hit_distance"]:
                video_query_parts.append(f'HitDistance >= {rec["hit_distance"] - 2} AND HitDistance <= {rec["hit_distance"] + 2}')
            video_query = ' AND '.join(video_query_parts)
            try:
                resp = requests.post(
                    "http://localhost:8000/mlb/video_search",
                    json={"query": video_query, "limit": 1}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("content", [])
                    if content and isinstance(content, list) and content[0].get("resource", {}).get("uri"):
                        rec["mlb_video_url"] = content[0]["resource"]["uri"]
                    else:
                        if mlb_video_url:
                            rec["mlb_video_url"] = mlb_video_url
                else:
                    if mlb_video_url:
                        rec["mlb_video_url"] = mlb_video_url
            except Exception as e:
                print(f"Error finding video for {rec['batter_name']} on {rec['date']}: {e}")
                if mlb_video_url:
                    rec["mlb_video_url"] = mlb_video_url
            records.append(rec)
        if len(records) > 3:
            return {"content": records}
        else:
            return {"content": records}
    except Exception as e:
        print(f"Error in statcast_leaderboard: {e}")
        return {"content": [], "error": str(e)}

# --- FastAPI endpoint for statcast leaderboard ---
@app.get("/statcast/leaderboard")
def statcast_leaderboard_endpoint(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    limit: int = Query(10, description="Number of results to return"),
    min_ev: float = Query(None, description="Minimum exit velocity (optional)"),
    result: str = Query(None, description="Filter by result type, e.g., 'Home Run' (optional)")
):
    return statcast_leaderboard(
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        min_ev=min_ev,
        result=result
    )
