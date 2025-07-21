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
  - `sort_by` (optional): Sort metric - "exit_velocity" (default), "distance", or "launch_angle"
  - `limit` (optional): Number of results (default: 10)
  - `order` (optional): Sort order - "desc" (default) or "asc"
- **Returns:** Detailed play-by-play data with video links

### 🎥 Video Highlights Integration

Every result from `statcast_leaderboard` includes multiple video access points:

```json
"video_links": {
  "game_highlights_url": "https://www.mlb.com/gameday/745890/video",
  "film_room_search": "https://www.mlb.com/video/search?q=Ronald+Acuna+Jr.+2024-07-20",
  "game_pk": "745890",
  "api_highlights_endpoint": "https://statsapi.mlb.com/api/v1/schedule?gamePk=745890&hydrate=game(content(highlights(highlights)))"
}
```

### 🏟️ Smart Team Recognition

The server intelligently handles team names:
- Full names: "Orioles" → "BAL", "Red Sox" → "BOS"
- Cities: "Boston" → "BOS", "New York Yankees" → "NYY"
- Historical teams: "Expos" → "MON", "Indians" → "CLE"
- All 30 current MLB teams supported with common variations

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