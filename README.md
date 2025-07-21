# MLB Stats MCP Server

A Model Context Protocol (MCP) server that provides comprehensive MLB baseball statistics with video highlights through the [pybaseball](https://github.com/jldbc/pybaseball) library.

## Overview

This MCP server enables AI assistants to access real-time MLB statistics, historical data, and video highlights. It provides four powerful tools for querying baseball data, from individual player performance to team statistics and advanced Statcast metrics.

## Features

### 🎯 4 Core Tools

#### 1. `get_player_stats`
Get detailed Statcast data for any MLB player with optional date filtering.
- **Parameters:**
  - `name` (required): Player name (e.g., "Mike Trout", "Ronald Acuña Jr.")
  - `start_date` (optional): Start date in YYYY-MM-DD format
  - `end_date` (optional): End date in YYYY-MM-DD format
- **Returns:** Player statistics including hits, home runs, exit velocity, launch angle, and barrel rate

#### 2. `get_team_stats`
Retrieve comprehensive team batting or pitching statistics for any season.
- **Parameters:**
  - `team` (required): Team name or abbreviation (e.g., "Yankees", "NYY", "Red Sox")
  - `year` (required): Season year (1871-present)
  - `stat_type` (optional): "batting" (default) or "pitching"
- **Returns:** Complete team statistics for the specified season

#### 3. `get_leaderboard`
Access statistical leaderboards for any MLB stat category.
- **Parameters:**
  - `stat` (required): Statistic abbreviation (e.g., "HR", "AVG", "ERA", "K")
  - `season` (required): Season year
  - `leaderboard_type` (optional): "batting" or "pitching"
  - `limit` (optional): Number of results (default: 10)
- **Returns:** Top players for the specified statistic

#### 4. `statcast_leaderboard`
Query advanced Statcast data with filtering, sorting, and video highlight links.
- **Parameters:**
  - `start_date` (required): Start date in YYYY-MM-DD format
  - `end_date` (required): End date in YYYY-MM-DD format
  - `result` (optional): Filter by outcome (e.g., "home_run", "single", "double")
  - `min_ev` (optional): Minimum exit velocity filter
  - `min_pitch_velo` (optional): Minimum pitch velocity filter
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

### Deploy on Smithery

1. Fork this repository
2. Connect your GitHub account to [Smithery](https://smithery.ai)
3. Deploy directly from your repository

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/mlb_mcp.git
cd mlb_mcp/san-juan

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m server
```

### Claude Desktop Configuration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "mlb-stats": {
      "command": "python",
      "args": ["-m", "server"],
      "cwd": "/path/to/mlb_mcp/san-juan"
    }
  }
}
```

## Usage Examples

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

### Team-Wide Rankings
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

## Technical Details

### Architecture
- **Transport**: stdio (Model Context Protocol)
- **Framework**: FastMCP for protocol implementation
- **Data Source**: pybaseball library (MLB official data)
- **Language**: Python 3.11+
- **Deployment**: Docker container via Smithery

### Key Features
- **Lazy Loading**: Heavy dependencies loaded only when needed for fast startup
- **Error Handling**: Comprehensive error messages for common issues
- **Type Safety**: Proper type conversion for JSON serialization
- **Performance**: Optimized for quick responses with data caching

### File Structure
```
san-juan/
├── server.py          # Main MCP server implementation
├── requirements.txt   # Python dependencies
├── smithery.yaml      # Smithery deployment configuration
├── Dockerfile         # Container configuration
└── README.md          # This file
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