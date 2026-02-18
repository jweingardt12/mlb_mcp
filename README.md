# MLB Stats MCP Server

A Model Context Protocol (MCP) server that provides comprehensive MLB baseball statistics with video highlights through the [pybaseball](https://github.com/jldbc/pybaseball) library.

## Overview

This MCP server enables AI assistants to access real-time MLB statistics, historical data, and video highlights. It provides eight powerful tools for querying baseball data, from individual player performance to team statistics and advanced Statcast metrics.

## Features

### 🎯 8 Core Tools

#### 1. `get_player_stats`
Get detailed Statcast data for any MLB player with optional date filtering.
- **Parameters:**
  - `name` (required): Player name (e.g., "Mike Trout", "Ronald Acuña Jr.")
  - `start_date` (optional): Start date in YYYY-MM-DD format
  - `end_date` (optional): End date in YYYY-MM-DD format
- **Returns:** Player statistics including hits, home runs, exit velocity, launch angle, and barrel rate

#### 2. `player_statcast`
Get comprehensive Statcast data for a specific player with advanced pitch-type and result filtering.
- **Parameters:**
  - `player_name` (required): Player name (e.g., "Aaron Judge", "Mike Trout")
  - `start_date` (optional): Start date in YYYY-MM-DD format (defaults to current season)
  - `end_date` (optional): End date in YYYY-MM-DD format (defaults to current season)
  - `pitch_type` (optional): Filter by pitch type - 'FF' (4-seam), 'SL' (slider), 'CH' (changeup), 'CU' (curveball), 'SI' (sinker), 'FC' (cutter), 'FS' (splitter)
  - `result_type` (optional): Filter by result - 'home_run', 'hit', 'single', 'double', 'triple', 'batted_ball'
  - `min_exit_velocity` (optional): Minimum exit velocity in mph
  - `min_distance` (optional): Minimum distance in feet
- **Returns:** Player stats with overall metrics, breakdown by pitch type/result, and top 5 examples with video links
- **Example Queries:**
  - "How many fastballs has Aaron Judge hit for a home run?"
  - "What's Shohei Ohtani's average exit velocity on sliders?"
  - "Show me Mike Trout's stats against breaking balls this season"

#### 3. `get_team_stats`
Retrieve comprehensive team batting or pitching statistics for any season.
- **Parameters:**
  - `team` (required): Team name or abbreviation (e.g., "Yankees", "NYY", "Red Sox")
  - `year` (required): Season year (1871-present)
  - `stat_type` (optional): "batting" (default) or "pitching"
- **Returns:** Complete team statistics for the specified season

#### 4. `get_leaderboard`
Access statistical leaderboards for any MLB stat category.
- **Parameters:**
  - `stat` (required): Statistic abbreviation (e.g., "HR", "AVG", "ERA", "K")
  - `season` (required): Season year
  - `leaderboard_type` (optional): "batting" or "pitching"
  - `limit` (optional): Number of results (default: 10)
- **Returns:** Top players for the specified statistic

#### 5. `statcast_leaderboard`
Query advanced Statcast data with filtering, sorting, and video highlight links.
- **Parameters:**
  - `start_date` (required): Start date in YYYY-MM-DD format
  - `end_date` (required): End date in YYYY-MM-DD format
  - `result` (optional): Filter by outcome (e.g., "home_run", "single", "double")
  - `min_ev` (optional): Minimum exit velocity filter
  - `min_pitch_velo` (optional): Minimum pitch velocity filter
  - `pitch_type` (optional): Filter by pitch type - 'FF' (4-seam), 'SL' (slider), 'CH' (changeup), 'CU' (curveball), 'SI' (sinker), 'FC' (cutter), 'FS' (splitter)
  - `player_name` (optional): Filter by batter name (e.g., "Aaron Judge", "Mike Trout")
  - `sort_by` (optional): Sort metric:
    - `"exit_velocity"` (default) - Hardest hit balls
    - `"distance"` - Longest hits
    - `"launch_angle"` - Optimal launch angles
    - `"pitch_velocity"` - Fastest pitches
    - `"spin_rate"` - Highest spin rates
    - `"xba"` - Highest expected batting average
    - `"xwoba"` - Highest expected weighted on-base average
    - `"barrel"` - Barrel rate (perfect contact)
  - `limit` (optional): Number of results (default: 10)
  - `order` (optional): Sort order - "desc" (default) or "asc"
  - `group_by` (optional): Group results by "team" for team-wide rankings
- **Returns:** Detailed play-by-play data with video links (or team aggregations when group_by="team")

#### 6. `team_season_stats`
Get fast team season averages for Statcast metrics. Optimized for queries like "which team hits the ball hardest?".
- **Parameters:**
  - `year` (required): Season year (e.g., 2025)
  - `stat` (optional): Metric to analyze
    - `"exit_velocity"` (default) - Average exit velocity
    - `"distance"` - Average and max distance
    - `"launch_angle"` - Average launch angle
    - `"barrel_rate"` - Percentage of barrels (perfect contact)
    - `"hard_hit_rate"` - Percentage of balls hit 95+ mph
    - `"sweet_spot_rate"` - Percentage hit at optimal launch angles (8-32°)
  - `min_result_type` (optional): Filter by result type
    - `"batted_ball"` - All balls in play
    - `"home_run"` - Home runs only
    - `"hit"` - All hits (singles, doubles, triples, home runs)
- **Returns:** Team rankings with averages, counts, and other statistics
- **Performance:** Uses sampling strategy (every 7th day) and 24-hour caching for instant responses

#### 7. `team_pitching_stats`
Get fast team pitching averages for Statcast metrics. Optimized for queries like "which team has the best pitching staff?".
- **Parameters:**
  - `year` (required): Season year (e.g., 2025)
  - `stat` (optional): Metric to analyze
    - `"velocity"` (default) - Average and max pitch velocity
    - `"spin_rate"` - Average and max spin rate
    - `"movement"` - Pitch break (horizontal, vertical, total)
    - `"whiff_rate"` - Swing-and-miss percentage
    - `"chase_rate"` - Swings at pitches outside the zone
    - `"zone_rate"` - Percentage of pitches in strike zone
    - `"ground_ball_rate"` - Ground balls per balls in play
    - `"xera"` - Expected ERA based on quality of contact
  - `pitch_type` (optional): Filter to specific pitch type
    - `"FF"` - 4-seam fastball
    - `"SL"` - Slider
    - `"CH"` - Changeup
    - `"CU"` - Curveball
    - `"SI"` - Sinker
    - `"FC"` - Cutter
    - `"FS"` - Splitter
- **Returns:** Team pitching rankings with averages, counts, and other statistics
- **Performance:** Uses sampling strategy (every 7th day) and 24-hour caching for instant responses

#### 8. `statcast_count`
Count Statcast events matching specific criteria. Optimized for multi-year queries like "how many 475+ ft home runs since 2023?"
- **Parameters:**
  - `start_date` (required): Start date in YYYY-MM-DD format
  - `end_date` (required): End date in YYYY-MM-DD format
  - `result_type` (optional): Type to count - 'home_run' (default), 'hit', 'batted_ball', or specific like 'double'
  - `player_name` (optional): Filter by batter name (e.g., "Aaron Judge", "Mike Trout")
  - `pitch_type` (optional): Filter by pitch type - 'FF' (4-seam), 'SL' (slider), 'CH' (changeup), 'CU' (curveball), 'SI' (sinker), 'FC' (cutter), 'FS' (splitter)
  - `min_distance` (optional): Minimum distance in feet (e.g., 475)
  - `max_distance` (optional): Maximum distance in feet
  - `min_exit_velocity` (optional): Minimum exit velocity in mph
  - `max_exit_velocity` (optional): Maximum exit velocity in mph
- **Returns:** Total count, yearly breakdown, and top 5 examples with video links
- **Performance:** 
  - Multi-year queries: Samples 3 days per month (~90x fewer API calls)
  - 6-12 months: Weekly sampling (~7x fewer API calls)
  - <6 months: Complete data with chunking
  - 24-hour caching for all queries

### 🎥 Video Highlights Integration

Every result from `statcast_leaderboard` includes detailed metrics and video access points:

```json
{
  "rank": 1,
  "player": "Ronald Acuña Jr.",
  "date": "2024-07-20",
  "exit_velocity": 113.7,
  "launch_angle": 23.0,
  "distance": 456.0,
  "result": "home_run",
  "pitch_velocity": 95.2,
  "pitch_type": "FF",
  "spin_rate": 2450,
  "xba": 0.920,
  "xwoba": 1.823,
  "barrel": true,
  "description": "Ronald Acuña Jr. homers (13) on a fly ball to center field.",
  "video_links": {
    "game_highlights_url": "https://www.mlb.com/gameday/745890/video",
    "film_room_search": "https://www.mlb.com/video/search?q=Ronald+Acuna+Jr.+2024-07-20",
    "game_pk": "745890",
    "api_highlights_endpoint": "https://statsapi.mlb.com/api/v1/schedule?gamePk=745890&hydrate=game(content(highlights(highlights)))"
  }
}
```

### 🏟️ Smart Team Recognition

The server intelligently handles team names:
- Full names: "Orioles" → "BAL", "Red Sox" → "BOS"
- Cities: "Boston" → "BOS", "New York Yankees" → "NYY"
- Historical teams: "Expos" → "MON", "Indians" → "CLE"
- All 30 current MLB teams supported with common variations

### 📊 Team-Wide Rankings

New team aggregation feature for `statcast_leaderboard`:
- Group results by team to see team-wide performance
- Calculates averages, maximums, and counts for each metric
- Perfect for questions like "Which team hits the hardest home runs?"
- Returns comprehensive team statistics including barrel counts and expected metrics

## Installation

### Deploy on Railway (Recommended for Smithery)

Deploy to Railway for a hosted HTTP MCP endpoint that Smithery can connect to:

1. Create a new Railway project and connect your GitHub repository
2. Railway will auto-detect the `Procfile` and deploy automatically
3. Your MCP endpoint will be available at `https://<your-railway-domain>/mcp`
4. Connect this URL to [Smithery](https://smithery.ai) for distribution

**Manual Start Command (if needed):**
If Railway doesn't auto-detect the Procfile, set this Start Command in Railway Settings:
```
uv run fastmcp run src/mlb_mcp/server.py:mcp --transport http --host 0.0.0.0 --port $PORT
```

### Deploy on Smithery Direct

1. Fork this repository
2. Connect your GitHub account to [Smithery](https://smithery.ai)
3. Deploy directly from your repository

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/mlb_mcp.git
cd mlb_mcp

# Install dependencies with uv
uv sync

# Run the server (stdio mode for local testing)
uv run python -m mlb_mcp.server

# Or run with HTTP transport for local HTTP testing
MCP_TRANSPORT=http MCP_PORT=8000 uv run python -m mlb_mcp.server
```

### Claude Desktop Configuration

Add to your Claude Desktop configuration (uses stdio transport):

```json
{
  "mcpServers": {
    "mlb-stats": {
      "command": "uv",
      "args": ["run", "python", "-m", "mlb_mcp.server"],
      "cwd": "/path/to/mlb_mcp"
    }
  }
}
```

## Usage Examples

### Player-Specific Queries (NEW!)

#### How many fastballs has Aaron Judge hit for a home run?
```
Query: "How many fastballs has Aaron Judge hit for a home run this season?"
Tool: player_statcast("Aaron Judge", "2024-04-01", "2024-10-31", "FF", "home_run")
OR
Tool: statcast_count("2024-04-01", "2024-10-31", "home_run", player_name="Aaron Judge", pitch_type="FF")
```

#### What's a player's performance on specific pitch types?
```
Query: "Show me Shohei Ohtani's stats against sliders this season"
Tool: player_statcast("Shohei Ohtani", "2024-04-01", "2024-10-31", "SL")
```

#### Player's hardest hit balls on specific pitch types
```
Query: "Show Aaron Judge's hardest hit balls on fastballs"
Tool: statcast_leaderboard("2024-04-01", "2024-10-31", player_name="Aaron Judge", pitch_type="FF", sort_by="exit_velocity", limit=10)
```

### Find the Longest Home Runs
```
Query: "Show me the longest home runs from yesterday"
Tool: statcast_leaderboard("2024-07-20", "2024-07-20", "home_run", None, None, "distance", 10)
```

### Hardest Hit Balls on Fast Pitches
```
Query: "What were the hardest hit balls on 99+ mph pitches last week?"
Tool: statcast_leaderboard("2024-07-14", "2024-07-20", None, 0, 99.0, "exit_velocity", 10)
```

### Fastest Pitches Thrown
```
Query: "Show me the fastest pitches thrown yesterday"
Tool: statcast_leaderboard("2024-07-20", "2024-07-20", None, None, None, "pitch_velocity", 10)
```

### Highest Spin Rate Pitches
```
Query: "What pitches had the highest spin rate today?"
Tool: statcast_leaderboard("2024-07-20", "2024-07-20", None, None, None, "spin_rate", 10)
```

### Best Quality Contact (Barrels)
```
Query: "Show me the best quality contact this week"
Tool: statcast_leaderboard("2024-07-14", "2024-07-20", None, None, None, "barrel", 10)
```

### Player Season Stats
```
Query: "Get Mike Trout's stats for this season"
Tool: get_player_stats("Mike Trout", "2024-04-01", "2024-10-01")
```

### Team Performance
```
Query: "How are the Red Sox doing this year?"
Tool: get_team_stats("Red Sox", 2024, "batting")
```

### League Leaders
```
Query: "Who's leading the league in home runs?"
Tool: get_leaderboard("HR", 2024, "batting", 10)
```

### Team-Wide Rankings (statcast_leaderboard)
```
Query: "Which team has the hardest hit home runs this season?"
Tool: statcast_leaderboard("2024-04-01", "2024-10-01", "home_run", None, None, "exit_velocity", 10, "desc", "team")
```

```
Query: "Show me teams with the longest average home run distance this month"
Tool: statcast_leaderboard("2024-07-01", "2024-07-31", "home_run", None, None, "distance", 10, "desc", "team")
```

```
Query: "Which teams have the highest average pitch velocity?"
Tool: statcast_leaderboard("2024-07-20", "2024-07-20", None, None, None, "pitch_velocity", 10, "desc", "team")
```

### Team Season Averages (team_season_stats)
```
Query: "What team averages the hardest hit balls in 2025?"
Tool: team_season_stats(2025, "exit_velocity")
```

```
Query: "Which team has the highest barrel rate this season?"
Tool: team_season_stats(2025, "barrel_rate")
```

```
Query: "Show me teams with the highest hard-hit rate on home runs only"
Tool: team_season_stats(2025, "hard_hit_rate", "home_run")
```

```
Query: "What team hits the ball the farthest on average?"
Tool: team_season_stats(2025, "distance")
```

### Team Pitching Analysis (team_pitching_stats)
```
Query: "Which team throws the hardest in 2025?"
Tool: team_pitching_stats(2025, "velocity")
```

```
Query: "What team has the best slider spin rate?"
Tool: team_pitching_stats(2025, "spin_rate", "SL")
```

```
Query: "Which pitching staff gets the most swings and misses?"
Tool: team_pitching_stats(2025, "whiff_rate")
```

```
Query: "Show me teams with the highest ground ball rate"
Tool: team_pitching_stats(2025, "ground_ball_rate")
```

```
Query: "Which team has the lowest expected ERA based on contact quality?"
Tool: team_pitching_stats(2025, "xera")
```

### Counting Queries (statcast_count)
```
Query: "How many home runs hit over 475 ft have been hit since 2023?"
Tool: statcast_count("2023-01-01", "2025-12-31", "home_run", 475)
```

```
Query: "Count all 110+ mph batted balls this season"
Tool: statcast_count("2025-04-01", "2025-10-31", "batted_ball", None, 110)
```

```
Query: "How many home runs between 400-450 feet were hit last year?"
Tool: statcast_count("2024-04-01", "2024-10-31", "home_run", 400, None, 450)
```

```
Query: "Total hits with exit velocity over 100 mph since 2022"
Tool: statcast_count("2022-01-01", "2025-12-31", "hit", None, 100)
```

## Technical Details

### Architecture
- **Transport**: stdio (local) or HTTP (Railway/hosted) - auto-detected via `PORT` env var
- **Framework**: FastMCP for protocol implementation
- **Data Source**: pybaseball library (MLB official data)
- **Language**: Python 3.11+
- **Deployment**: Railway (HTTP) or Smithery (containerized)

### Performance Optimizations
- **Query Chunking**: Automatically splits large date ranges into 5-day chunks to handle Baseball Savant's 30,000 row limit
- **Response Caching**: 15-minute cache for repeated queries, 24-hour cache for team season stats
- **Vectorized Operations**: Uses NumPy for efficient team identification instead of slower pandas apply() operations
- **Sampling Strategy**: 
  - `team_season_stats` and `team_pitching_stats`: Every 7th day sampling
  - `statcast_count`: Adaptive sampling (3 days/month for multi-year, weekly for 6-12 months)
- **Specialized Tools**: Dedicated tools for common aggregate queries that would timeout with full data:
  - `team_season_stats` for batting metrics
  - `team_pitching_stats` for pitching metrics
  - `statcast_count` for counting queries across multiple years
- **Lazy Loading**: Heavy dependencies (pandas, numpy, pybaseball) loaded only when needed for fast startup
- **Efficient Filtering**: Applies filters sequentially to minimize data processing overhead

### Key Features
- **Error Handling**: Comprehensive error messages for common issues
- **Type Safety**: Proper type conversion for JSON serialization
- **Video Integration**: Automatic video highlight links for all plays and team top performances
- **Smart Team Lookup**: Handles full names, abbreviations, cities, and historical teams

### File Structure
```
mlb_mcp/
├── src/
│   └── mlb_mcp/
│       ├── __init__.py    # Package exports
│       └── server.py      # Main MCP server implementation
├── pyproject.toml         # Project configuration and dependencies
├── smithery.yaml          # Smithery deployment configuration
├── Procfile               # Railway start command
└── README.md              # This file
```

## Requirements

- Python 3.11 or higher
- Internet connection for MLB data access
- Smithery account for deployment (optional)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project uses publicly available MLB data through the pybaseball library. All MLB data is property of MLB Advanced Media.

---

**Built with** [pybaseball](https://github.com/jldbc/pybaseball) and [FastMCP](https://github.com/jlowin/fastmcp)