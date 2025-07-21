#!/usr/bin/env python3
"""MLB Stats MCP Server using fastmcp for Smithery deployment"""

from fastmcp import FastMCP
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP("mlb-stats-mcp")

# Lazy loading for pybaseballstats
pybaseball = None

def load_pybaseball():
    """Lazy load pybaseballstats to avoid startup delays"""
    global pybaseball
    if pybaseball is None:
        try:
            import pybaseballstats as pb
            pybaseball = pb
            logger.info("Successfully loaded pybaseballstats")
        except ImportError:
            try:
                import pybaseball as pb
                pybaseball = pb
                logger.info("Successfully loaded pybaseball")
            except ImportError:
                logger.error("Could not import pybaseballstats or pybaseball")
                raise
    return pybaseball

@mcp.tool()
async def get_player_stats(name: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Get player statcast data by name (optionally filter by date range: YYYY-MM-DD)
    
    Args:
        name: Player name to search for
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        JSON string of player statistics
    """
    try:
        pb = load_pybaseball()
        
        # Search for player
        results = pb.playerid_lookup(name.split()[-1], name.split()[0] if len(name.split()) > 1 else '')
        if results.empty:
            return f"No player found matching '{name}'"
        
        player_id = results.iloc[0]['key_fangraphs']
        
        # Get statcast data
        import pandas as pd
        if start_date and end_date:
            data = pb.statcast_batter(start_date, end_date, player_id)
        else:
            # Default to current season
            from datetime import datetime
            current_year = datetime.now().year
            data = pb.statcast_batter(f"{current_year}-04-01", f"{current_year}-10-01", player_id)
        
        if data.empty:
            return f"No statcast data found for {name}"
        
        # Format results
        stats = {
            "player": name,
            "games": len(data['game_date'].unique()),
            "at_bats": len(data),
            "hits": len(data[data['events'] == 'single']) + len(data[data['events'] == 'double']) + 
                    len(data[data['events'] == 'triple']) + len(data[data['events'] == 'home_run']),
            "home_runs": len(data[data['events'] == 'home_run']),
            "avg_exit_velocity": data['launch_speed'].mean(),
            "avg_launch_angle": data['launch_angle'].mean(),
            "barrel_rate": (len(data[data['barrel'] == 1]) / len(data) * 100) if 'barrel' in data.columns else None
        }
        
        import json
        return json.dumps(stats, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting player stats: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_team_stats(team: str, year: int, stat_type: str = "batting") -> str:
    """
    Get team stats for a given team and year. Type can be 'batting' or 'pitching'
    
    Args:
        team: Team name or abbreviation
        year: Year/season to get stats for
        stat_type: Type of stats - 'batting' or 'pitching'
    
    Returns:
        JSON string of team statistics
    """
    try:
        pb = load_pybaseball()
        
        # Get team stats
        if stat_type.lower() == "batting":
            data = pb.team_batting(year)
        elif stat_type.lower() == "pitching":
            data = pb.team_pitching(year)
        else:
            return f"Invalid stat_type: {stat_type}. Use 'batting' or 'pitching'"
        
        # Find the team
        import pandas as pd
        team_upper = team.upper()
        team_data = data[data['Team'].str.contains(team_upper, case=False, na=False)]
        
        if team_data.empty:
            return f"No data found for team '{team}' in {year}"
        
        # Convert to dict and format
        result = team_data.iloc[0].to_dict()
        
        # Clean up NaN values
        import numpy as np
        import json
        for key, value in result.items():
            if isinstance(value, float) and np.isnan(value):
                result[key] = None
                
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting team stats: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def get_leaderboard(stat: str, season: int, leaderboard_type: str = "batting", limit: int = 10) -> str:
    """
    Get leaderboard for a given stat and season
    
    Args:
        stat: Statistic to get leaderboard for (e.g., 'HR', 'AVG', 'ERA')
        season: Season year to get leaderboard for
        leaderboard_type: Type of leaderboard - 'batting' or 'pitching'
        limit: Number of results to return (default 10)
    
    Returns:
        JSON string of leaderboard data
    """
    try:
        pb = load_pybaseball()
        
        # Get the appropriate leaderboard
        if leaderboard_type.lower() == "batting":
            data = pb.batting_stats(season)
        elif leaderboard_type.lower() == "pitching":
            data = pb.pitching_stats(season)
        else:
            return f"Invalid leaderboard_type: {leaderboard_type}. Use 'batting' or 'pitching'"
        
        # Check if stat exists
        if stat not in data.columns:
            return f"Stat '{stat}' not found. Available stats: {', '.join(data.columns)}"
        
        # Sort by the stat and get top players
        import pandas as pd
        sorted_data = data.sort_values(by=stat, ascending=False).head(limit)
        
        # Create leaderboard
        leaderboard = []
        for idx, row in sorted_data.iterrows():
            entry = {
                "rank": idx + 1,
                "player": row.get('Name', 'Unknown'),
                "team": row.get('Team', 'Unknown'),
                stat: row[stat]
            }
            leaderboard.append(entry)
        
        import json
        return json.dumps({
            "stat": stat,
            "season": season,
            "type": leaderboard_type,
            "leaderboard": leaderboard
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting leaderboard: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def statcast_leaderboard(start_date: str, end_date: str, result: Optional[str] = None, 
                             min_ev: Optional[float] = None, limit: int = 10, order: str = "desc") -> str:
    """
    Get event-level Statcast leaderboard for a date range, filtered by result (e.g., home run) 
    and sorted by exit velocity, etc.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        result: Filter by result type, e.g., 'home_run' (optional)
        min_ev: Minimum exit velocity (optional)
        limit: Number of results to return
        order: Sort order - 'asc' or 'desc'
    
    Returns:
        JSON string of statcast leaderboard
    """
    try:
        pb = load_pybaseball()
        
        # Get statcast data for date range
        data = pb.statcast(start_dt=start_date, end_dt=end_date)
        
        if data.empty:
            return f"No statcast data found for date range {start_date} to {end_date}"
        
        # Filter by result if specified
        if result:
            data = data[data['events'] == result]
            if data.empty:
                return f"No {result} events found in the specified date range"
        
        # Filter by minimum exit velocity
        if min_ev:
            data = data[data['launch_speed'] >= min_ev]
        
        # Sort by exit velocity
        ascending = (order.lower() == 'asc')
        sorted_data = data.sort_values(by='launch_speed', ascending=ascending).head(limit)
        
        # Create leaderboard
        leaderboard = []
        for idx, row in sorted_data.iterrows():
            entry = {
                "rank": idx + 1,
                "player": row.get('player_name', 'Unknown'),
                "date": row.get('game_date', 'Unknown'),
                "exit_velocity": row.get('launch_speed', None),
                "launch_angle": row.get('launch_angle', None),
                "distance": row.get('hit_distance_sc', None),
                "result": row.get('events', None),
                "description": row.get('des', None)
            }
            leaderboard.append(entry)
        
        import json
        return json.dumps({
            "start_date": start_date,
            "end_date": end_date,
            "filter": {"result": result, "min_ev": min_ev},
            "leaderboard": leaderboard
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting statcast leaderboard: {str(e)}")
        return f"Error: {str(e)}"

def main():
    """Main entry point for the MCP server"""
    logger.info("Starting MLB Stats MCP server")
    mcp.run()

if __name__ == "__main__":
    main()