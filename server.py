#!/usr/bin/env python3
"""MLB Stats MCP Server using fastmcp for Smithery deployment"""

from fastmcp import FastMCP
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta
import hashlib
import json
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP("mlb-stats-mcp")

# Lazy loading for pybaseball
pybaseball = None

# Cache for statcast queries (15 minute TTL)
query_cache = {}

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

def chunk_date_range(start_date: str, end_date: str, max_days: int = 5) -> List[tuple]:
    """Split date range into chunks to avoid API limits"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    chunks = []
    
    while start <= end:
        chunk_end = min(start + timedelta(days=max_days - 1), end)
        chunks.append((start.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
        start = chunk_end + timedelta(days=1)
    
    return chunks

def get_cache_key(start_date: str, end_date: str, **kwargs) -> str:
    """Generate cache key for query"""
    key_data = f"{start_date}_{end_date}_{json.dumps(kwargs, sort_keys=True)}"
    return hashlib.md5(key_data.encode()).hexdigest()

def is_cache_valid(cache_entry: dict) -> bool:
    """Check if cache entry is still valid (15 minute TTL)"""
    if not cache_entry:
        return False
    cached_time = datetime.fromisoformat(cache_entry['timestamp'])
    return (datetime.now() - cached_time).seconds < 900

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
async def team_season_stats(year: int, stat: str = "exit_velocity", min_result_type: Optional[str] = None) -> str:
    """
    Get team season averages for Statcast metrics. Optimized for fast team comparisons.
    
    Args:
        year: Season year (e.g., 2025)
        stat: Metric to analyze - 'exit_velocity', 'distance', 'launch_angle', 'barrel_rate', 
              'hard_hit_rate' (95+ mph), 'sweet_spot_rate' (8-32 degree launch angle)
        min_result_type: Filter to specific results like 'batted_ball' (all), 'home_run', 'hit' (optional)
    
    Returns:
        JSON string of team rankings by selected metric
    """
    try:
        # Special cache for season-wide team stats (24-hour TTL)
        cache_key = f"team_season_{year}_{stat}_{min_result_type}"
        
        if cache_key in query_cache:
            cached_time = datetime.fromisoformat(query_cache[cache_key]['timestamp'])
            if (datetime.now() - cached_time).seconds < 86400:  # 24 hour cache
                logger.info(f"Using cached team season data for {cache_key}")
                return query_cache[cache_key]['data']
        
        pb = load_pybaseball()
        from pybaseball import statcast
        import pandas as pd
        import numpy as np
        
        # For current year, use year-to-date. For past years, use April-October
        if year == datetime.now().year:
            start_date = f"{year}-04-01"
            end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            start_date = f"{year}-04-01" 
            end_date = f"{year}-10-31"
        
        logger.info(f"Fetching team season stats for {year}, stat={stat}")
        
        # Use a sampling approach for current season to avoid timeouts
        # Sample every 7th day to get representative data quickly
        sample_dates = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            sample_dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=7)
        
        # Fetch sample data
        all_data = []
        for sample_date in sample_dates:
            try:
                # Get one day of data
                data = statcast(start_dt=sample_date, end_dt=sample_date)
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                logger.error(f"Error fetching data for {sample_date}: {str(e)}")
                continue
        
        if not all_data:
            return f"No data available for {year} season"
        
        # Combine samples
        data = pd.concat(all_data, ignore_index=True)
        
        # Filter by result type if specified
        if min_result_type:
            if min_result_type == 'batted_ball':
                data = data[data['type'] == 'X']  # Balls in play
            elif min_result_type == 'hit':
                data = data[data['events'].isin(['single', 'double', 'triple', 'home_run'])]
            elif min_result_type == 'home_run':
                data = data[data['events'] == 'home_run']
            else:
                data = data[data['events'] == min_result_type]
        
        # Add batting team using vectorized operations
        data['batting_team'] = np.where(
            data['inning_topbot'] == 'Top',
            data['away_team'],
            data['home_team']
        )
        
        # Calculate team statistics based on requested stat
        if stat == 'exit_velocity':
            team_stats = data.groupby('batting_team')['launch_speed'].agg(['mean', 'count', 'std']).round(2)
            team_stats.columns = ['avg_exit_velocity', 'batted_balls', 'std_dev']
        elif stat == 'distance':
            team_stats = data.groupby('batting_team')['hit_distance_sc'].agg(['mean', 'max', 'count']).round(2)
            team_stats.columns = ['avg_distance', 'max_distance', 'batted_balls']
        elif stat == 'launch_angle':
            team_stats = data.groupby('batting_team')['launch_angle'].agg(['mean', 'count']).round(2)
            team_stats.columns = ['avg_launch_angle', 'batted_balls']
        elif stat == 'barrel_rate':
            barrel_stats = data.groupby('batting_team').agg({
                'barrel': lambda x: (x == 1).sum(),
                'launch_speed': 'count'
            })
            team_stats = pd.DataFrame({
                'barrel_rate': (barrel_stats['barrel'] / barrel_stats['launch_speed'] * 100).round(2),
                'barrels': barrel_stats['barrel'],
                'batted_balls': barrel_stats['launch_speed']
            })
        elif stat == 'hard_hit_rate':
            hard_hit = data[data['launch_speed'] >= 95].groupby('batting_team').size()
            total = data.groupby('batting_team')['launch_speed'].count()
            team_stats = pd.DataFrame({
                'hard_hit_rate': (hard_hit / total * 100).round(2),
                'hard_hit_balls': hard_hit,
                'batted_balls': total
            })
        elif stat == 'sweet_spot_rate':
            sweet_spot = data[(data['launch_angle'] >= 8) & (data['launch_angle'] <= 32)].groupby('batting_team').size()
            total = data.groupby('batting_team')['launch_angle'].count()
            team_stats = pd.DataFrame({
                'sweet_spot_rate': (sweet_spot / total * 100).round(2),
                'sweet_spot_balls': sweet_spot,
                'batted_balls': total
            })
        
        # Sort by primary metric
        primary_col = team_stats.columns[0]
        team_stats = team_stats.sort_values(by=primary_col, ascending=False)
        
        # Create leaderboard
        leaderboard = []
        for idx, (team, row) in enumerate(team_stats.iterrows()):
            entry = {
                "rank": idx + 1,
                "team": team,
                **row.to_dict()
            }
            leaderboard.append(entry)
        
        response = json.dumps({
            "year": year,
            "stat": stat,
            "filter": min_result_type,
            "sample_size": f"{len(sample_dates)} days sampled",
            "total_events": len(data),
            "leaderboard": leaderboard
        }, indent=2, default=str)
        
        # Cache the result
        query_cache[cache_key] = {
            'data': response,
            'timestamp': datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting team season stats: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def team_pitching_stats(year: int, stat: str = "velocity", pitch_type: Optional[str] = None) -> str:
    """
    Get team pitching averages for Statcast metrics. Optimized for fast team pitching comparisons.
    
    Args:
        year: Season year (e.g., 2025)
        stat: Metric to analyze - 'velocity', 'spin_rate', 'movement', 'whiff_rate', 
              'chase_rate', 'zone_rate', 'ground_ball_rate', 'xera' (expected ERA)
        pitch_type: Filter to specific pitch type - 'FF' (4-seam), 'SL' (slider), 'CH' (changeup), 
                   'CU' (curveball), 'SI' (sinker), 'FC' (cutter), 'FS' (splitter) (optional)
    
    Returns:
        JSON string of team pitching rankings by selected metric
    """
    try:
        # Special cache for season-wide pitching stats (24-hour TTL)
        cache_key = f"team_pitching_{year}_{stat}_{pitch_type}"
        
        if cache_key in query_cache:
            cached_time = datetime.fromisoformat(query_cache[cache_key]['timestamp'])
            if (datetime.now() - cached_time).seconds < 86400:  # 24 hour cache
                logger.info(f"Using cached team pitching data for {cache_key}")
                return query_cache[cache_key]['data']
        
        pb = load_pybaseball()
        from pybaseball import statcast
        import pandas as pd
        import numpy as np
        
        # For current year, use year-to-date. For past years, use April-October
        if year == datetime.now().year:
            start_date = f"{year}-04-01"
            end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            start_date = f"{year}-04-01" 
            end_date = f"{year}-10-31"
        
        logger.info(f"Fetching team pitching stats for {year}, stat={stat}")
        
        # Use sampling approach (every 7th day)
        sample_dates = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            sample_dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=7)
        
        # Fetch sample data
        all_data = []
        for sample_date in sample_dates:
            try:
                data = statcast(start_dt=sample_date, end_dt=sample_date)
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                logger.error(f"Error fetching data for {sample_date}: {str(e)}")
                continue
        
        if not all_data:
            return f"No data available for {year} season"
        
        # Combine samples
        data = pd.concat(all_data, ignore_index=True)
        
        # Filter by pitch type if specified
        if pitch_type:
            data = data[data['pitch_type'] == pitch_type]
            if data.empty:
                return f"No data found for pitch type {pitch_type}"
        
        # Add pitching team using vectorized operations
        data['pitching_team'] = np.where(
            data['inning_topbot'] == 'Top',
            data['home_team'],
            data['away_team']
        )
        
        # Calculate team statistics based on requested stat
        if stat == 'velocity':
            team_stats = data.groupby('pitching_team')['release_speed'].agg(['mean', 'max', 'count']).round(2)
            team_stats.columns = ['avg_velocity', 'max_velocity', 'pitches']
        elif stat == 'spin_rate':
            team_stats = data.groupby('pitching_team')['release_spin_rate'].agg(['mean', 'max', 'count']).round(2)
            team_stats.columns = ['avg_spin_rate', 'max_spin_rate', 'pitches']
        elif stat == 'movement':
            # Calculate total movement as sqrt(pfx_x^2 + pfx_z^2)
            data['total_movement'] = np.sqrt(data['pfx_x']**2 + data['pfx_z']**2)
            team_stats = data.groupby('pitching_team').agg({
                'pfx_x': 'mean',
                'pfx_z': 'mean',
                'total_movement': ['mean', 'count']
            }).round(2)
            team_stats.columns = ['avg_horizontal_break', 'avg_vertical_break', 'avg_total_movement', 'pitches']
        elif stat == 'whiff_rate':
            # Whiffs are swinging strikes
            swings = data[data['description'].isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 
                                                    'foul_tip', 'hit_into_play', 'hit_into_play_no_out', 
                                                    'hit_into_play_score'])]
            whiffs = swings[swings['description'].isin(['swinging_strike', 'swinging_strike_blocked'])]
            whiff_stats = whiffs.groupby('pitching_team').size()
            swing_stats = swings.groupby('pitching_team').size()
            team_stats = pd.DataFrame({
                'whiff_rate': (whiff_stats / swing_stats * 100).round(2),
                'whiffs': whiff_stats,
                'swings': swing_stats
            })
        elif stat == 'chase_rate':
            # Chase rate: swings at pitches outside the zone
            out_of_zone = data[(data['plate_x'].abs() > 0.83) | (data['plate_z'] < 1.5) | (data['plate_z'] > 3.5)]
            chases = out_of_zone[out_of_zone['description'].isin(['swinging_strike', 'swinging_strike_blocked', 
                                                                   'foul', 'foul_tip', 'hit_into_play'])]
            chase_count = chases.groupby('pitching_team').size()
            out_zone_count = out_of_zone.groupby('pitching_team').size()
            team_stats = pd.DataFrame({
                'chase_rate': (chase_count / out_zone_count * 100).round(2),
                'chases': chase_count,
                'pitches_out_of_zone': out_zone_count
            })
        elif stat == 'zone_rate':
            # Zone rate: percentage of pitches in the strike zone
            in_zone = data[(data['plate_x'].abs() <= 0.83) & (data['plate_z'] >= 1.5) & (data['plate_z'] <= 3.5)]
            zone_count = in_zone.groupby('pitching_team').size()
            total_count = data.groupby('pitching_team').size()
            team_stats = pd.DataFrame({
                'zone_rate': (zone_count / total_count * 100).round(2),
                'pitches_in_zone': zone_count,
                'total_pitches': total_count
            })
        elif stat == 'ground_ball_rate':
            # Ground ball rate on balls in play
            in_play = data[data['type'] == 'X']  # Balls in play
            ground_balls = in_play[in_play['bb_type'] == 'ground_ball']
            gb_count = ground_balls.groupby('pitching_team').size()
            in_play_count = in_play.groupby('pitching_team').size()
            team_stats = pd.DataFrame({
                'ground_ball_rate': (gb_count / in_play_count * 100).round(2),
                'ground_balls': gb_count,
                'balls_in_play': in_play_count
            })
        elif stat == 'xera':
            # Expected ERA based on quality of contact allowed
            team_stats = data.groupby('pitching_team').agg({
                'estimated_woba_using_speedangle': 'mean',
                'launch_speed': 'mean',
                'launch_angle': 'mean',
                'release_speed': 'count'
            }).round(3)
            # Convert xwOBA to approximate xERA (rough formula: xERA ≈ 13 * xwOBA - 2)
            team_stats['xera'] = (13 * team_stats['estimated_woba_using_speedangle'] - 2).round(2)
            team_stats.columns = ['avg_xwoba', 'avg_exit_velocity_against', 'avg_launch_angle_against', 
                                  'pitches', 'expected_era']
        
        # Sort by primary metric (lower is better for most pitching stats)
        primary_col = team_stats.columns[0]
        ascending = stat in ['whiff_rate', 'zone_rate', 'ground_ball_rate', 'velocity', 'spin_rate', 'movement']
        team_stats = team_stats.sort_values(by=primary_col, ascending=not ascending)
        
        # Create leaderboard
        leaderboard = []
        for idx, (team, row) in enumerate(team_stats.iterrows()):
            entry = {
                "rank": idx + 1,
                "team": team,
                **row.to_dict()
            }
            leaderboard.append(entry)
        
        response = json.dumps({
            "year": year,
            "stat": stat,
            "pitch_type_filter": pitch_type,
            "sample_size": f"{len(sample_dates)} days sampled",
            "total_pitches": len(data),
            "leaderboard": leaderboard
        }, indent=2, default=str)
        
        # Cache the result
        query_cache[cache_key] = {
            'data': response,
            'timestamp': datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting team pitching stats: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def statcast_count(start_date: str, end_date: str, result_type: str = "home_run", 
                        min_distance: Optional[float] = None, min_exit_velocity: Optional[float] = None,
                        max_distance: Optional[float] = None, max_exit_velocity: Optional[float] = None) -> str:
    """
    Count Statcast events matching criteria. Optimized for multi-year queries like "how many 475+ ft home runs since 2023?"
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format  
        result_type: Type of result to count - 'home_run' (default), 'hit', 'batted_ball', or specific like 'double'
        min_distance: Minimum distance in feet (e.g., 475 for long home runs)
        min_exit_velocity: Minimum exit velocity in mph
        max_distance: Maximum distance in feet
        max_exit_velocity: Maximum exit velocity in mph
    
    Returns:
        JSON with count, breakdown by year, and top examples
    """
    try:
        # Special cache for counting queries (24-hour TTL)
        cache_key = f"count_{start_date}_{end_date}_{result_type}_{min_distance}_{min_exit_velocity}_{max_distance}_{max_exit_velocity}"
        
        if cache_key in query_cache:
            cached_time = datetime.fromisoformat(query_cache[cache_key]['timestamp'])
            if (datetime.now() - cached_time).seconds < 86400:  # 24 hour cache
                logger.info(f"Using cached count data for {cache_key}")
                return query_cache[cache_key]['data']
        
        pb = load_pybaseball()
        from pybaseball import statcast
        import pandas as pd
        import numpy as np
        
        logger.info(f"Counting {result_type} from {start_date} to {end_date}")
        
        # For multi-year queries, use monthly sampling to avoid timeouts
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        days_span = (end - start).days
        
        # Determine sampling strategy based on date range
        if days_span > 365:
            # For multi-year, sample 3 days per month
            sample_days = [1, 15, 28]
        elif days_span > 180:
            # For 6-12 months, sample weekly
            sample_days = list(range(1, 32, 7))
        else:
            # For < 6 months, use regular chunking
            date_chunks = chunk_date_range(start_date, end_date)
            all_data = []
            
            for chunk_start, chunk_end in date_chunks:
                try:
                    data = statcast(start_dt=chunk_start, end_dt=chunk_end)
                    if not data.empty:
                        all_data.append(data)
                except Exception as e:
                    logger.error(f"Error fetching chunk {chunk_start} to {chunk_end}: {str(e)}")
                    continue
            
            if all_data:
                data = pd.concat(all_data, ignore_index=True)
            else:
                data = pd.DataFrame()
        
        # For long date ranges, use sampling
        if days_span > 180:
            all_data = []
            current = start
            
            while current <= end:
                year = current.year
                month = current.month
                
                for day in sample_days:
                    try:
                        # Create valid date
                        sample_date = datetime(year, month, min(day, 28 if month == 2 else 30 if month in [4,6,9,11] else 31))
                        if sample_date > end:
                            break
                        if sample_date < start:
                            continue
                            
                        date_str = sample_date.strftime('%Y-%m-%d')
                        logger.info(f"Sampling {date_str}")
                        
                        sample_data = statcast(start_dt=date_str, end_dt=date_str)
                        if not sample_data.empty:
                            all_data.append(sample_data)
                    except Exception as e:
                        logger.error(f"Error sampling {year}-{month}-{day}: {str(e)}")
                        continue
                
                # Move to next month
                if month == 12:
                    current = datetime(year + 1, 1, 1)
                else:
                    current = datetime(year, month + 1, 1)
            
            if all_data:
                data = pd.concat(all_data, ignore_index=True)
            else:
                data = pd.DataFrame()
        
        if data.empty:
            return json.dumps({"error": "No data found for the specified date range"})
        
        # Apply filters
        if result_type == 'home_run':
            data = data[data['events'] == 'home_run']
        elif result_type == 'hit':
            data = data[data['events'].isin(['single', 'double', 'triple', 'home_run'])]
        elif result_type == 'batted_ball':
            data = data[data['type'] == 'X']
        else:
            data = data[data['events'] == result_type]
        
        if min_distance:
            data = data[data['hit_distance_sc'] >= min_distance]
        if max_distance:
            data = data[data['hit_distance_sc'] <= max_distance]
        if min_exit_velocity:
            data = data[data['launch_speed'] >= min_exit_velocity]
        if max_exit_velocity:
            data = data[data['launch_speed'] <= max_exit_velocity]
        
        # Calculate scaling factor for sampled data
        if days_span > 180:
            # Estimate total based on sampling
            sample_size = len(all_data)
            if days_span > 365:
                # 3 days per month sampling
                months_span = days_span / 30.4
                expected_samples = months_span * 3
                scaling_factor = days_span / expected_samples
            else:
                # Weekly sampling
                scaling_factor = 7
        else:
            scaling_factor = 1
        
        # Get counts
        total_count = len(data)
        estimated_total = int(total_count * scaling_factor)
        
        # Breakdown by year
        data['year'] = pd.to_datetime(data['game_date']).dt.year
        yearly_counts = data.groupby('year').size()
        yearly_breakdown = {
            str(year): int(count * scaling_factor) 
            for year, count in yearly_counts.items()
        }
        
        # Get top examples
        examples = []
        if not data.empty:
            # Sort by distance for home runs, exit velocity for others
            sort_col = 'hit_distance_sc' if result_type == 'home_run' and 'hit_distance_sc' in data.columns else 'launch_speed'
            top_data = data.nlargest(5, sort_col)
            
            for _, row in top_data.iterrows():
                example = {
                    'player': str(row.get('player_name', 'Unknown')),
                    'date': str(row.get('game_date', 'Unknown')),
                    'distance': float(row.get('hit_distance_sc')) if pd.notna(row.get('hit_distance_sc')) else None,
                    'exit_velocity': float(row.get('launch_speed')) if pd.notna(row.get('launch_speed')) else None,
                    'team': str(row.get('batting_team', 'Unknown'))
                }
                
                # Add video links if available
                game_pk = row.get('game_pk')
                if game_pk:
                    example['video_url'] = f"https://www.mlb.com/gameday/{game_pk}/video"
                
                examples.append(example)
        
        # Create response
        response = {
            'start_date': start_date,
            'end_date': end_date,
            'filters': {
                'result_type': result_type,
                'min_distance': min_distance,
                'max_distance': max_distance,
                'min_exit_velocity': min_exit_velocity,
                'max_exit_velocity': max_exit_velocity
            },
            'count': estimated_total,
            'actual_sampled': total_count,
            'sampling_note': f"Estimated from {sample_size} sampled days" if days_span > 180 else "Complete data",
            'yearly_breakdown': yearly_breakdown,
            'top_examples': examples
        }
        
        response_json = json.dumps(response, indent=2, default=str)
        
        # Cache the result
        query_cache[cache_key] = {
            'data': response_json,
            'timestamp': datetime.now().isoformat()
        }
        
        return response_json
        
    except Exception as e:
        logger.error(f"Error in statcast_count: {str(e)}")
        return json.dumps({"error": str(e)})

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
        # Check cache first
        cache_key = get_cache_key(start_date, end_date, result=result, min_ev=min_ev, 
                                 min_pitch_velo=min_pitch_velo, sort_by=sort_by, 
                                 limit=limit, order=order, group_by=group_by)
        
        if cache_key in query_cache and is_cache_valid(query_cache[cache_key]):
            logger.info(f"Using cached data for query {cache_key}")
            return query_cache[cache_key]['data']
        
        pb = load_pybaseball()
        
        # Import statcast function and pandas
        from pybaseball import statcast
        import pandas as pd
        import numpy as np
        
        logger.info(f"Fetching statcast data for {start_date} to {end_date}")
        
        # Split into chunks for large date ranges
        date_chunks = chunk_date_range(start_date, end_date)
        all_data = []
        
        for chunk_start, chunk_end in date_chunks:
            try:
                logger.info(f"Fetching chunk: {chunk_start} to {chunk_end}")
                chunk_data = statcast(start_dt=chunk_start, end_dt=chunk_end)
                if not chunk_data.empty:
                    all_data.append(chunk_data)
            except Exception as e:
                logger.error(f"Error fetching chunk {chunk_start} to {chunk_end}: {str(e)}")
                continue
        
        if not all_data:
            return f"No statcast data found for date range {start_date} to {end_date}"
        
        # Combine all chunks
        data = pd.concat(all_data, ignore_index=True) if len(all_data) > 1 else all_data[0]
        
        # Apply filters efficiently
        if result:
            data = data[data['events'] == result]
            if data.empty:
                return f"No {result} events found in the specified date range"
        
        if min_ev:
            data = data[data['launch_speed'] >= min_ev]
        
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
        
        # Vectorized team identification (much faster than apply)
        if 'inning_topbot' in data.columns and 'home_team' in data.columns and 'away_team' in data.columns:
            data['batting_team'] = np.where(
                data['inning_topbot'] == 'Top',
                data['away_team'],
                data['home_team']
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
            
            # Create team leaderboard with top highlights
            leaderboard = []
            for idx, (team, row) in enumerate(team_stats.iterrows()):
                # Get top play for this team
                team_data = data[data['batting_team'] == team]
                if not team_data.empty:
                    # Find the best play for this team based on sort criteria
                    top_play = team_data.nlargest(1, sort_column).iloc[0]
                    
                    # Generate video links for the top play
                    game_pk = top_play.get('game_pk')
                    top_play_video = {}
                    if game_pk:
                        player_name = str(top_play.get('player_name', '')).replace(' ', '+')
                        game_date = str(top_play.get('game_date', ''))
                        top_play_video = {
                            'player': str(top_play.get('player_name', 'Unknown')),
                            'date': game_date,
                            'game_highlights_url': f"https://www.mlb.com/gameday/{game_pk}/video",
                            'film_room_search': f"https://www.mlb.com/video/search?q={player_name}+{game_date}" if player_name and game_date else None,
                            'description': str(top_play.get('des', 'No description'))
                        }
                
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
                    "barrel_count": int(row.get('barrel_<lambda>', 0)) if 'barrel_<lambda>' in row else 0,
                    "top_play_video": top_play_video if top_play_video else None
                }
                leaderboard.append(entry)
            
            # Create response for team grouping
            response = json.dumps({
                "start_date": start_date,
                "end_date": end_date,
                "filter": {"result": result, "min_ev": min_ev, "min_pitch_velo": min_pitch_velo},
                "sorted_by": sort_by,
                "group_by": "team",
                "leaderboard": leaderboard
            }, indent=2, default=str)
            
            # Cache the result
            query_cache[cache_key] = {
                'data': response,
                'timestamp': datetime.now().isoformat()
            }
            
            return response
        
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
            
            # Always generate comprehensive video links when game_pk is available
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
        
        # Create response
        response = json.dumps({
            "start_date": start_date,
            "end_date": end_date,
            "filter": {"result": result, "min_ev": min_ev, "min_pitch_velo": min_pitch_velo},
            "sorted_by": sort_by,
            "leaderboard": leaderboard
        }, indent=2, default=str)
        
        # Cache the result
        query_cache[cache_key] = {
            'data': response,
            'timestamp': datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting statcast leaderboard: {str(e)}")
        return f"Error: {str(e)}"

def main():
    """Main entry point for the MCP server"""
    logger.info("Starting MLB Stats MCP server")
    mcp.run()

if __name__ == "__main__":
    main()