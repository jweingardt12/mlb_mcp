# MLB Stats MCP Server

A Model Context Protocol (MCP) server that provides MLB baseball statistics through the [pybaseball](https://github.com/jldbc/pybaseball) library.

## Features

This server exposes 4 tools for accessing MLB data:

1. **get_player_stats** - Get player statcast data by name with optional date filtering
   - Parameters: `name` (required), `start_date`, `end_date` (optional, YYYY-MM-DD format)
   
2. **get_team_stats** - Get team batting or pitching statistics for a given year
   - Parameters: `team` (required), `year` (required), `stat_type` (optional, defaults to "batting")
   
3. **get_leaderboard** - Get statistical leaderboards (HR, AVG, ERA, etc.)
   - Parameters: `stat` (required), `season` (required), `leaderboard_type` (optional), `limit` (optional)
   
4. **statcast_leaderboard** - Get event-level Statcast data filtered by exit velocity, pitch velocity, result type, etc.
   - Parameters: `start_date` (required), `end_date` (required), `result`, `min_ev`, `min_pitch_velo`, `limit`, `order` (optional)

## Deployment

This server is designed for deployment on [Smithery](https://smithery.ai) and uses stdio transport for communication.

### Configuration

The server uses the Model Context Protocol with stdio transport. See `smithery.yaml` for deployment configuration.

### Requirements

- Python 3.11+
- fastmcp library for MCP protocol support
- pybaseball for MLB data access
- pandas, numpy, and other dependencies (installed automatically)

## Installation

### Smithery
Deploy directly to Smithery by connecting your GitHub repository.

### Local Development
```bash
pip install -r requirements.txt
python -m server
```

Note: This is an MCP server designed to be used with MCP clients like Claude Desktop or through Smithery's interface.

## Usage Examples

Once deployed and connected to an MCP client, you can use the tools like:

```
# Get player statistics
get_player_stats("Mike Trout", "2024-04-01", "2024-10-01")

# Get team batting stats
get_team_stats("NYY", 2024, "batting")

# Get home run leaderboard
get_leaderboard("HR", 2024, "batting", 10)

# Get hardest hit balls in July 2024
statcast_leaderboard("2024-07-01", "2024-07-31", "home_run", 95.0, None, 10)

# Get hardest hit balls on 99+ mph pitches
statcast_leaderboard("2024-07-01", "2024-07-31", None, 0, 99.0, 10)
```

## Architecture

- `server.py` - Main MCP server implementation using fastmcp
- `smithery.yaml` - Smithery deployment configuration
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration for Smithery deployment

The server uses lazy loading for heavy dependencies (pandas, numpy) to ensure fast startup times and avoid timeouts during tool discovery.

---

**Powered by [pybaseball](https://github.com/jldbc/pybaseball) and [fastmcp](https://github.com/jlowin/fastmcp)**