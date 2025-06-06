# Pybaseball MCP Server


This is a FastAPI-based MCP server that exposes MLB and Fangraphs baseball data via the [pybaseballstats](https://pypi.org/project/pybaseballstats/) library.
The `pybaseballstats` package relies on common scientific Python libraries such as pandas and numpy.

## Features
- `/player?name=...` — Get player Statcast data by name (optionally filter by date range)
- `/team_stats?team=...&year=...&type=batting|pitching` — Get team batting or pitching stats for a given year
- `/leaderboard?stat=...&season=...&league=...&type=batting|pitching` — Get MLB leaderboard for a stat, season, and league

## Setup
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Example Usage
- **Player stats:**
  - `GET /player?name=Mike Trout`
  - `GET /player?name=Shohei Ohtani&start_date=2023-04-01&end_date=2023-10-01`
- **Team stats:**
  - `GET /team_stats?team=Yankees&year=2023&type=batting`
  - `GET /team_stats?team=Dodgers&year=2022&type=pitching`
- **Leaderboards:**
  - `GET /leaderboard?stat=HR&season=2023&league=AL&type=batting`
  - `GET /leaderboard?stat=ERA&season=2023&league=NL&type=pitching`

## API Documentation
One up and running, interactive API docs are available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Publishing
To expose an MCP STDIO interface instead of HTTP, first start the FastAPI server and then run the wrapper:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 &
python mcp_stdio_wrapper.py
```

This project is ready for deployment on Smithery or any other MCP-compatible platform. It uses the `pybaseballstats` library and its dependencies, including pandas and numpy.

## Listing available tools
The server supports the MCP tool-discovery protocol. After starting the server, you can list available tools using either HTTP or the STDIO wrapper.

**HTTP**
```bash
curl -X POST http://localhost:8000/tools/list
```

**STDIO**
```bash
echo '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}' | python mcp_stdio_wrapper.py
```


---

**Powered by [pybaseballstats](https://pypi.org/project/pybaseballstats/) and FastAPI**
