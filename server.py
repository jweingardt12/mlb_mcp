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

# Lazy loading for pybaseball
pybaseball = None

def load_pybaseball():
    """Lazy load pybaseball to avoid startup delays"""
    global pybaseball
    if pybaseball is None:
        try:
            import pybaseball as pb
            pybaseball = pb
            logger.info("Successfully loaded pybaseball")
            # Log version if available
            if hasattr(pb, '__version__'):
                logger.info(f"pybaseball version: {pb.__version__}")
        except ImportError as e:
            logger.error(f"Could not import pybaseball: {str(e)}")
            logger.error("Make sure pybaseball is installed: pip install pybaseball")
            raise ImportError("pybaseball library is not installed. Please install it with: pip install pybaseball")
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
        
        # Import functions from pybaseball
        from pybaseball import playerid_lookup, statcast_batter
        
        # Search for player
        last_name = name.split()[-1]
        first_name = name.split()[0] if len(name.split()) > 1 else ''
        results = playerid_lookup(last_name, first_name)
        if results.empty:
            return f"No player found matching '{name}'"
        
        player_id = results.iloc[0]['key_fangraphs']
        
        # Get statcast data
        import pandas as pd
        if start_date and end_date:
            data = statcast_batter(start_date, end_date, player_id)
        else:
            # Default to current season
            from datetime import datetime
            current_year = datetime.now().year
            data = statcast_batter(f"{current_year}-04-01", f"{current_year}-10-01", player_id)
        
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
        
        # Import functions from pybaseball
        from pybaseball import team_batting, team_pitching
        
        # Validate year
        from datetime import datetime
        current_year = datetime.now().year
        if year < 1871 or year > current_year:
            return f"Invalid year: {year}. Please use a year between 1871 and {current_year}"
        
        # Get team stats
        logger.info(f"Fetching {stat_type} stats for {year}")
        
        try:
            if stat_type.lower() == "batting":
                data = team_batting(year)
            elif stat_type.lower() == "pitching":
                data = team_pitching(year)
            else:
                return f"Invalid stat_type: {stat_type}. Use 'batting' or 'pitching'"
        except Exception as e:
            logger.error(f"Error fetching team stats: {str(e)}")
            return f"Error fetching {stat_type} stats for {year}: {str(e)}"
        
        # Team name to abbreviation mapping
        team_mapping = {
            'orioles': 'BAL', 'baltimore': 'BAL',
            'red sox': 'BOS', 'boston': 'BOS',
            'yankees': 'NYY', 'new york yankees': 'NYY',
            'rays': 'TBR', 'tampa': 'TBR', 'tampa bay': 'TBR',
            'blue jays': 'TOR', 'toronto': 'TOR',
            'white sox': 'CHW', 'chicago white sox': 'CHW',
            'guardians': 'CLE', 'cleveland': 'CLE', 'indians': 'CLE',
            'tigers': 'DET', 'detroit': 'DET',
            'royals': 'KCR', 'kansas city': 'KCR',
            'twins': 'MIN', 'minnesota': 'MIN',
            'astros': 'HOU', 'houston': 'HOU',
            'angels': 'LAA', 'los angeles angels': 'LAA',
            'athletics': 'OAK', 'oakland': 'OAK', 'as': 'OAK',
            'mariners': 'SEA', 'seattle': 'SEA',
            'rangers': 'TEX', 'texas': 'TEX',
            'braves': 'ATL', 'atlanta': 'ATL',
            'marlins': 'MIA', 'miami': 'MIA', 'florida': 'FLA',
            'mets': 'NYM', 'new york mets': 'NYM',
            'phillies': 'PHI', 'philadelphia': 'PHI',
            'nationals': 'WSN', 'washington': 'WSN', 'expos': 'MON',
            'cubs': 'CHC', 'chicago cubs': 'CHC',
            'reds': 'CIN', 'cincinnati': 'CIN',
            'brewers': 'MIL', 'milwaukee': 'MIL',
            'pirates': 'PIT', 'pittsburgh': 'PIT',
            'cardinals': 'STL', 'st louis': 'STL', 'st. louis': 'STL',
            'diamondbacks': 'ARI', 'arizona': 'ARI', 'dbacks': 'ARI',
            'rockies': 'COL', 'colorado': 'COL',
            'dodgers': 'LAD', 'los angeles dodgers': 'LAD',
            'padres': 'SDP', 'san diego': 'SDP',
            'giants': 'SFG', 'san francisco': 'SFG'
        }
        
        # Find the team
        import pandas as pd
        
        # Try to find team abbreviation
        team_lower = team.lower()
        team_abbr = team_mapping.get(team_lower, team.upper())
        
        # Try exact match first
        team_data = data[data['Team'] == team_abbr]
        
        # If not found, try contains search
        if team_data.empty:
            team_data = data[data['Team'].str.contains(team_abbr, case=False, na=False)]
        
        # If still not found, try original team name
        if team_data.empty:
            team_data = data[data['Team'].str.contains(team, case=False, na=False)]
        
        if team_data.empty:
            # List available teams
            available_teams = data['Team'].unique().tolist()
            return f"No data found for team '{team}' in {year}. Available teams: {', '.join(sorted(available_teams))}"
        
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
        
        # Import functions from pybaseball
        from pybaseball import batting_stats, pitching_stats
        
        # Get the appropriate leaderboard
        if leaderboard_type.lower() == "batting":
            data = batting_stats(season)
        elif leaderboard_type.lower() == "pitching":
            data = pitching_stats(season)
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
                             min_ev: Optional[float] = None, min_pitch_velo: Optional[float] = None,
                             sort_by: str = "exit_velocity", limit: int = 10, order: str = "desc",
                             group_by: Optional[str] = None) -> str:
    """
    Get event-level Statcast leaderboard for a date range with advanced filtering and sorting.
    
    Supports sorting by exit velocity, distance, pitch velocity, spin rate, expected stats, and more.
    Includes video highlight links for each result.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        result: Filter by result type, e.g., 'home_run' (optional)
        min_ev: Minimum exit velocity (optional)
        min_pitch_velo: Minimum pitch velocity in mph (optional)
        sort_by: Metric to sort by - 'exit_velocity', 'distance', 'launch_angle', 'pitch_velocity', 
                'spin_rate', 'xba', 'xwoba', 'barrel' (default: 'exit_velocity')
        limit: Number of results to return
        order: Sort order - 'asc' or 'desc'
        group_by: Group results by 'team' for team-wide rankings (optional)
    
    Returns:
        JSON string of statcast leaderboard
    """
    try:
        pb = load_pybaseball()
        
        # Import statcast function from pybaseball
        from pybaseball import statcast
        
        logger.info(f"Fetching statcast data for {start_date} to {end_date}")
        
        # Get statcast data for date range
        try:
            data = statcast(start_dt=start_date, end_dt=end_date)
        except Exception as e:
            logger.error(f"Error fetching statcast data: {str(e)}")
            return f"Error fetching data: {str(e)}. This might be due to network issues or invalid dates."
        
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
        
        # Filter by minimum pitch velocity
        if min_pitch_velo:
            data = data[data['release_speed'] >= min_pitch_velo]
            if data.empty:
                return f"No pitches found with velocity >= {min_pitch_velo} mph in the specified criteria"
        
        # Determine sort column
        sort_column_map = {
            'exit_velocity': 'launch_speed',
            'distance': 'hit_distance_sc',
            'launch_angle': 'launch_angle',
            'pitch_velocity': 'release_speed',
            'spin_rate': 'release_spin_rate',
            'xba': 'estimated_ba_using_speedangle',
            'xwoba': 'estimated_woba_using_speedangle',
            'barrel': 'barrel'
        }
        sort_column = sort_column_map.get(sort_by, 'launch_speed')
        
        # Add team information for batting events
        # For batting events, the batter's team is determined by whether they're home or away
        # This requires checking the inning_topbot field
        if 'inning_topbot' in data.columns and 'home_team' in data.columns and 'away_team' in data.columns:
            data['batting_team'] = data.apply(
                lambda row: row['away_team'] if row['inning_topbot'] == 'Top' else row['home_team'],
                axis=1
            )
        else:
            data['batting_team'] = 'Unknown'
        
        # Handle team grouping if requested
        if group_by == 'team':
            # Group by team and calculate aggregates
            import pandas as pd
            import numpy as np
            
            # Remove rows with null values in sort column before grouping
            data = data.dropna(subset=[sort_column])
            
            if data.empty:
                return f"No data available for sorting by {sort_by}"
            
            # Group by team and calculate statistics
            team_stats = data.groupby('batting_team').agg({
                sort_column: ['mean', 'max', 'count'],
                'launch_speed': 'mean',
                'hit_distance_sc': 'mean',
                'release_speed': 'mean',
                'release_spin_rate': 'mean',
                'estimated_ba_using_speedangle': 'mean',
                'estimated_woba_using_speedangle': 'mean',
                'barrel': lambda x: (x == 1).sum() if 'barrel' in data.columns else 0
            }).round(2)
            
            # Flatten column names
            team_stats.columns = ['_'.join(col).strip() for col in team_stats.columns.values]
            
            # Sort by the main metric
            sort_col_mean = f"{sort_column}_mean"
            sort_col_max = f"{sort_column}_max"
            sort_col_for_team = sort_col_max if sort_by in ['distance', 'exit_velocity', 'pitch_velocity'] else sort_col_mean
            
            team_stats = team_stats.sort_values(by=sort_col_for_team, ascending=(order.lower() == 'asc')).head(limit)
            
            # Create team leaderboard
            leaderboard = []
            for idx, (team, row) in enumerate(team_stats.iterrows()):
                entry = {
                    "rank": idx + 1,
                    "team": team,
                    "count": int(row.get(f"{sort_column}_count", 0)),
                    f"{sort_by}_avg": float(row.get(f"{sort_column}_mean", 0)),
                    f"{sort_by}_max": float(row.get(f"{sort_column}_max", 0)),
                    "avg_exit_velocity": float(row.get('launch_speed_mean', 0)) if 'launch_speed_mean' in row else None,
                    "avg_distance": float(row.get('hit_distance_sc_mean', 0)) if 'hit_distance_sc_mean' in row else None,
                    "avg_pitch_velocity": float(row.get('release_speed_mean', 0)) if 'release_speed_mean' in row else None,
                    "avg_spin_rate": float(row.get('release_spin_rate_mean', 0)) if 'release_spin_rate_mean' in row else None,
                    "avg_xba": float(row.get('estimated_ba_using_speedangle_mean', 0)) if 'estimated_ba_using_speedangle_mean' in row else None,
                    "avg_xwoba": float(row.get('estimated_woba_using_speedangle_mean', 0)) if 'estimated_woba_using_speedangle_mean' in row else None,
                    "barrel_count": int(row.get('barrel_<lambda>', 0)) if 'barrel_<lambda>' in row else 0
                }
                leaderboard.append(entry)
            
            import json
            return json.dumps({
                "start_date": start_date,
                "end_date": end_date,
                "filter": {"result": result, "min_ev": min_ev, "min_pitch_velo": min_pitch_velo},
                "sorted_by": sort_by,
                "group_by": "team",
                "leaderboard": leaderboard
            }, indent=2, default=str)
        
        # Remove rows with null values in sort column
        data = data.dropna(subset=[sort_column])
        
        if data.empty:
            return f"No data available for sorting by {sort_by}"
        
        # Sort by specified metric
        ascending = (order.lower() == 'asc')
        sorted_data = data.sort_values(by=sort_column, ascending=ascending).head(limit)
        
        # Create leaderboard
        leaderboard = []
        for idx, row in sorted_data.iterrows():
            # Get game_pk for video links
            game_pk = row.get('game_pk')
            
            # Generate video-related URLs
            video_info = {}
            if game_pk:
                # MLB.com game highlights page
                video_info['game_highlights_url'] = f"https://www.mlb.com/gameday/{game_pk}/video"
                
                # MLB Film Room search URL for this player and date
                player_name = str(row.get('player_name', '')).replace(' ', '+')
                game_date = str(row.get('game_date', ''))
                if player_name and game_date:
                    video_info['film_room_search'] = f"https://www.mlb.com/video/search?q={player_name}+{game_date}"
                
                # Include game_pk for API access
                video_info['game_pk'] = str(game_pk)
                
                # MLB Stats API endpoint for game highlights
                video_info['api_highlights_endpoint'] = f"https://statsapi.mlb.com/api/v1/schedule?gamePk={game_pk}&hydrate=game(content(highlights(highlights)))"
            
            entry = {
                "rank": idx + 1,
                "player": str(row.get('player_name', 'Unknown')),
                "date": str(row.get('game_date', 'Unknown')),
                "exit_velocity": float(row.get('launch_speed')) if row.get('launch_speed') is not None else None,
                "launch_angle": float(row.get('launch_angle')) if row.get('launch_angle') is not None else None,
                "distance": float(row.get('hit_distance_sc')) if row.get('hit_distance_sc') is not None else None,
                "result": str(row.get('events', 'Unknown')),
                "pitch_velocity": float(row.get('release_speed')) if row.get('release_speed') is not None else None,
                "pitch_type": str(row.get('pitch_type', 'Unknown')),
                "spin_rate": float(row.get('release_spin_rate')) if row.get('release_spin_rate') is not None else None,
                "xba": float(row.get('estimated_ba_using_speedangle')) if row.get('estimated_ba_using_speedangle') is not None else None,
                "xwoba": float(row.get('estimated_woba_using_speedangle')) if row.get('estimated_woba_using_speedangle') is not None else None,
                "barrel": bool(row.get('barrel') == 1) if row.get('barrel') is not None else None,
                "team": str(row.get('batting_team', 'Unknown')),
                "description": str(row.get('des', 'No description')),
                "video_links": video_info if video_info else None
            }
            leaderboard.append(entry)
        
        import json
        return json.dumps({
            "start_date": start_date,
            "end_date": end_date,
            "filter": {"result": result, "min_ev": min_ev, "min_pitch_velo": min_pitch_velo},
            "sorted_by": sort_by,
            "leaderboard": leaderboard
        }, indent=2, default=str)  # default=str handles any remaining type issues
        
    except Exception as e:
        logger.error(f"Error getting statcast leaderboard: {str(e)}")
        return f"Error: {str(e)}"

def main():
    """Main entry point for the MCP server"""
    logger.info("Starting MLB Stats MCP server")
    mcp.run()

if __name__ == "__main__":
    main()