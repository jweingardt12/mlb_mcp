# MLB MCP Server - Changelog

## Version 2.0.0 - Player & Pitch Type Filtering

### 🎯 Major Enhancements

#### New Tool: `player_statcast`
A comprehensive player-specific Statcast query tool that can answer questions like:
- "How many fastballs has Aaron Judge hit for a home run?"
- "What's Shohei Ohtani's average exit velocity on sliders?"
- "Show me Mike Trout's stats against breaking balls this season"

**Parameters:**
- `player_name` (required): Player name (e.g., "Aaron Judge", "Mike Trout")
- `start_date` (optional): Start date in YYYY-MM-DD format (defaults to current season)
- `end_date` (optional): End date in YYYY-MM-DD format (defaults to current season)
- `pitch_type` (optional): Filter by pitch type - 'FF' (4-seam), 'SL' (slider), 'CH' (changeup), 'CU' (curveball), 'SI' (sinker), 'FC' (cutter), 'FS' (splitter)
- `result_type` (optional): Filter by result - 'home_run', 'hit', 'single', 'double', 'triple', 'batted_ball'
- `min_exit_velocity` (optional): Minimum exit velocity in mph
- `min_distance` (optional): Minimum distance in feet

**Returns:**
- Overall stats (avg/max exit velocity, distance, launch angle, barrel rate)
- Breakdown by pitch type (if not filtered)
- Breakdown by result type (if not filtered)
- Top 5 examples with video links

#### Enhanced: `statcast_count`
Now supports player and pitch type filtering!

**New Parameters:**
- `player_name` (optional): Filter by batter name (e.g., "Aaron Judge", "Mike Trout")
- `pitch_type` (optional): Filter by pitch type - 'FF' (4-seam), 'SL' (slider), 'CH' (changeup), 'CU' (curveball), 'SI' (sinker), 'FC' (cutter), 'FS' (splitter)

**Example Queries:**
- "How many fastballs has Aaron Judge hit for a home run?"
  ```
  statcast_count("2024-04-01", "2024-10-31", "home_run", player_name="Aaron Judge", pitch_type="FF")
  ```
- "Count all sliders hit for doubles by Mike Trout this year"
  ```
  statcast_count("2024-04-01", "2024-10-31", "double", player_name="Mike Trout", pitch_type="SL")
  ```

#### Enhanced: `statcast_leaderboard`
Now supports player and pitch type filtering for leaderboards!

**New Parameters:**
- `player_name` (optional): Filter by batter name (e.g., "Aaron Judge", "Mike Trout")
- `pitch_type` (optional): Filter by pitch type - 'FF' (4-seam), 'SL' (slider), 'CH' (changeup), 'CU' (curveball), 'SI' (sinker), 'FC' (cutter), 'FS' (splitter)

**Example Queries:**
- "Show Aaron Judge's hardest hit balls on fastballs"
  ```
  statcast_leaderboard("2024-04-01", "2024-10-31", player_name="Aaron Judge", pitch_type="FF", sort_by="exit_velocity")
  ```
- "Show team rankings for exit velocity on sliders"
  ```
  statcast_leaderboard("2024-04-01", "2024-10-31", pitch_type="SL", sort_by="exit_velocity", group_by="team")
  ```

### 🐛 Bug Fixes

#### Fixed: Player ID Lookup for Statcast Queries
- **Issue**: `get_player_stats` was using Fangraphs ID instead of MLBAM ID for statcast queries
- **Fix**: Now correctly uses MLBAM ID (key_mlbam) for all statcast queries
- **Impact**: Player lookups now work correctly with pybaseball's statcast functions

### 📊 New Query Capabilities

The server can now answer these types of questions:

1. **Player + Pitch Type + Result**
   - "How many fastballs has Aaron Judge hit for a home run?"
   - "Show me Shohei Ohtani's home runs on breaking balls"

2. **Player Performance by Pitch Type**
   - "What's Aaron Judge's average exit velocity on fastballs vs sliders?"
   - "How does Mike Trout perform against changeups?"

3. **Counting Queries with Filters**
   - "Count all home runs on 4-seam fastballs in 2024"
   - "How many sliders were hit for home runs by the Yankees?"

4. **Leaderboards with Advanced Filtering**
   - "Show the hardest hit fastballs by Aaron Judge this season"
   - "Which teams hit sliders the hardest?"

### 🔧 Technical Improvements

1. **Consistent Player ID Handling**: All tools now use MLBAM ID for statcast queries
2. **Efficient Filtering**: Player and pitch type filters applied early in the pipeline to reduce memory usage
3. **Better Error Messages**: Improved error handling with helpful suggestions when players aren't found
4. **Cache Keys Updated**: Cache keys now include player_id and pitch_type for proper caching

### 📝 Updated Documentation

- Added `player_statcast` tool to README
- Updated `statcast_count` documentation with new parameters
- Updated `statcast_leaderboard` documentation with new parameters
- Added example queries for all new functionality

---

## Migration Guide

### For Existing Users

All existing queries continue to work without changes. The new parameters are optional.

### New Query Patterns

**Before:** Could not filter by player or pitch type
```
# This was not possible before
❌ "How many fastballs has Aaron Judge hit for a home run?"
```

**Now:** Full support for player and pitch type filtering
```
✅ statcast_count("2024-04-01", "2024-10-31", "home_run", player_name="Aaron Judge", pitch_type="FF")
✅ player_statcast("Aaron Judge", pitch_type="FF", result_type="home_run")
✅ statcast_leaderboard("2024-04-01", "2024-10-31", player_name="Aaron Judge", pitch_type="FF")
```

---

**Upgrade Impact:** 🟢 Low - All changes are backwards compatible
