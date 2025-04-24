# Pybaseball MCP Server

This is a FastAPI-based MCP server that exposes MLB and Fangraphs baseball data via the [pybaseball](https://pypi.org/project/pybaseball/) library.

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
Interactive API docs are available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Publishing
This project is ready for deployment on Smithery or any other MCP-compatible platform. It uses only pure Python dependencies for easy deployment.

---

**Powered by [pybaseball](https://pypi.org/project/pybaseball/) and FastAPI**
