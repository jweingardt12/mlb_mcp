#!/usr/bin/env python3
"""MLB Stats MCP Server using fastmcp for Smithery deployment"""

import os
from fastmcp import FastMCP
from typing import Optional, Dict, Any, List, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import hashlib
import json
import time
import calendar
import re
from functools import lru_cache
from collections import OrderedDict
from smithery.decorators import smithery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy loading for pybaseball
pybaseball = None

# Cache for statcast queries (15 minute TTL)
query_cache = {}

# Cache for player names (permanent during server lifetime)
player_name_cache = {}

# Team alias mapping used across multiple tools
TEAM_ALIASES = {
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
    'athletics': 'OAK', 'oakland': 'OAK',
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
# Ensure abbreviations map to themselves
TEAM_ALIASES.update({abbr.lower(): abbr for abbr in set(TEAM_ALIASES.values())})

SEARCH_RESULT_CACHE_LIMIT = 256
search_result_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
SEARCH_CAPABILITY_KEY = "search"
SEARCH_CAPABILITY_VERSION = "2025-09-16"


def _store_search_result_cache(result_id: str, payload: Dict[str, Any]) -> None:
    """Store search metadata for later fetch requests with basic LRU pruning."""
    if not result_id:
        return
    search_result_cache[result_id] = payload
    search_result_cache.move_to_end(result_id)
    while len(search_result_cache) > SEARCH_RESULT_CACHE_LIMIT:
        search_result_cache.popitem(last=False)


def resolve_team_alias(team: str) -> str:
    """Resolve team names or abbreviations to MLB three-letter code."""
    if not team:
        return team
    return TEAM_ALIASES.get(team.lower(), team.upper())


def generate_search_candidates(query_text: str, limit_value: int) -> Tuple[int, List[Dict[str, Any]]]:
    """Generate connector-compliant search candidates for a natural language query.

    Returns:
        A tuple of (year_context, list_of_candidate_dicts)
    """
    results: List[Dict[str, Any]] = []
    normalized_query = query_text.lower().strip()
    normalized_words = set(re.findall(r"[a-z0-9/\\+]+", normalized_query))

    today = datetime.now()
    current_year = today.year
    year_matches = re.findall(r"\b(?:19|20)\d{2}\b", query_text)
    year_context = current_year
    if year_matches:
        try:
            year_context = int(year_matches[-1])
            year_context = max(1871, min(year_context, current_year))
        except ValueError:
            year_context = current_year

    default_start = f"{year_context}-03-01"
    default_end = today.strftime('%Y-%m-%d') if year_context >= current_year else f"{year_context}-11-15"

    def add_result(result_id: str, title: str, snippet: str, tool: str, parameters: Dict[str, Any],
                   score: float = 1.0, url: Optional[str] = None, metadata_extra: Optional[Dict[str, Any]] = None,
                   prepend: bool = False) -> None:
        metadata = {
            "tool": tool,
            "parameters": parameters
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        candidate = {
            "id": result_id,
            "title": title,
            "snippet": snippet,
            "score": score,
            "url": url,
            "metadata": metadata
        }
        if prepend:
            results.insert(0, candidate)
        else:
            results.append(candidate)

    # Player detection
    stop_words = {
        "statcast", "stats", "stat", "team", "teams", "leaderboard", "leaders",
        "search", "count", "season", "since", "how", "many", "latest", "compare",
        "against", "versus", "vs", "what", "is", "are", "the", "a", "an", "mlb",
        "exit", "velocity", "launch", "angle", "home", "runs", "for", "top", "best"
    }
    candidate_tokens = [token for token in re.findall(r"[A-Za-z']+", query_text) if token.lower() not in stop_words]

    try:
        if candidate_tokens:
            last_name = candidate_tokens[-1]
            first_name = " ".join(candidate_tokens[:-1])

            pb = load_pybaseball()
            from pybaseball import playerid_lookup

            player_matches = playerid_lookup(last_name, first_name)

            if player_matches.empty and len(candidate_tokens) >= 2:
                flipped_last = candidate_tokens[0]
                flipped_first = " ".join(candidate_tokens[1:])
                player_matches = playerid_lookup(flipped_last, flipped_first)

            for _, player in player_matches.head(limit_value).iterrows():
                raw_full_name = f"{player.get('name_first', '').strip()} {player.get('name_last', '').strip()}".strip()
                if not raw_full_name:
                    continue
                display_name = " ".join(part.capitalize() if part else part for part in raw_full_name.split())
                full_name = display_name or raw_full_name
                player_id = player.get('key_mlbam')
                snippet = f"Statcast overview for {full_name} from {default_start} through {default_end}."
                add_result(
                    result_id=f"player:{player_id}",
                    title=f"{full_name} Statcast overview",
                    snippet=snippet,
                    tool="get_player_stats",
                    parameters={
                        "name": full_name,
                        "start_date": None,
                        "end_date": None
                    },
                    metadata_extra={
                        "player_id": player_id,
                        "name": full_name
                    }
                )
                if len(results) >= limit_value:
                    break
    except Exception as exc:
        logger.debug(f"Search player lookup failed: {exc}")

    # Team detection
    if len(results) < limit_value:
        matched_team = None
        for alias, abbr in TEAM_ALIASES.items():
            if " " in alias:
                if alias in normalized_query:
                    matched_team = abbr
                    break
            else:
                if alias in normalized_words:
                    matched_team = abbr
                    break

        if matched_team:
            stat_type = "pitching" if "pitch" in normalized_query else "batting"
            snippet = f"{matched_team} {stat_type} totals for the {year_context} season."
            add_result(
                result_id=f"team:{matched_team}:{year_context}:{stat_type}",
                title=f"{matched_team} {year_context} {stat_type} stats",
                snippet=snippet,
                tool="get_team_stats",
                parameters={
                    "team": matched_team,
                    "year": year_context,
                    "stat_type": stat_type
                },
                metadata_extra={
                    "team": matched_team,
                    "year": year_context,
                    "stat_type": stat_type
                }
            )

    # Leaderboard detection
    if len(results) < limit_value:
        stat_keywords = {
            'home run': ('HR', 'batting'),
            'home runs': ('HR', 'batting'),
            'hr': ('HR', 'batting'),
            'avg': ('AVG', 'batting'),
            'average': ('AVG', 'batting'),
            'ops': ('OPS', 'batting'),
            'obp': ('OBP', 'batting'),
            'slg': ('SLG', 'batting'),
            'woba': ('wOBA', 'batting'),
            'era': ('ERA', 'pitching'),
            'whip': ('WHIP', 'pitching'),
            'strikeout': ('SO', 'pitching'),
            'strikeouts': ('SO', 'pitching'),
            'k/9': ('K/9', 'pitching'),
            'sv': ('SV', 'pitching')
        }
        leaderboard_match = None
        for keyword, mapped in stat_keywords.items():
            if keyword in normalized_query:
                leaderboard_match = mapped
                break

        if ('leaderboard' in normalized_query or 'leaders' in normalized_query or leaderboard_match) and len(results) < limit_value:
            stat, leaderboard_type = leaderboard_match if leaderboard_match else ('HR', 'batting')
            snippet = f"Top {leaderboard_type} performers for {stat} in {year_context}."
            add_result(
                result_id=f"leaderboard:{stat}:{leaderboard_type}:{year_context}",
                title=f"{year_context} {stat} {leaderboard_type} leaderboard",
                snippet=snippet,
                tool="get_leaderboard",
                parameters={
                    "stat": stat,
                    "season": year_context,
                    "leaderboard_type": leaderboard_type,
                    "limit": limit_value
                },
                metadata_extra={
                    "stat": stat,
                    "season": year_context,
                    "leaderboard_type": leaderboard_type
                }
            )

    # Statcast leaderboard / count heuristics
    fastball_query = 'fastball' in normalized_query or 'fastballs' in normalized_query
    if ('home run' in normalized_query or 'home runs' in normalized_query) and 'most' in normalized_query and fastball_query and any(token in normalized_query for token in ['95', '96', '97', '98', '99', '100']):
        velocity_threshold = 95.0
        try:
            numbers = re.findall(r"(\d{2,3})\s*mph", query_text.lower())
            if numbers:
                velocity_threshold = float(numbers[0])
        except Exception:
            pass

        snippet = f"Leaders in home runs on fastballs >= {velocity_threshold} mph for {year_context}."
        add_result(
            result_id=f"statcast_fastball_hr:{year_context}:{int(velocity_threshold)}",
            title=f"Fastball HR leaders {year_context}",
            snippet=snippet,
            tool="statcast_leaderboard",
            parameters={
                "start_date": default_start,
                "end_date": default_end,
                "result": "home_run",
                "min_pitch_velo": velocity_threshold,
                "pitch_type": "FASTBALL",
                "group_by": "player",
                "sort_by": "count",
                "limit": min(limit_value, 10)
            },
            prepend=True
        )

    if len(results) < limit_value and ("statcast" in normalized_query or "exit velocity" in normalized_query or "launch angle" in normalized_query):
        snippet = f"Detailed Statcast leaderboard from {default_start} to {default_end}."
        add_result(
            result_id=f"statcast_leaderboard:{year_context}",
            title=f"Statcast leaderboard {year_context}",
            snippet=snippet,
            tool="statcast_leaderboard",
            parameters={
                "start_date": default_start,
                "end_date": default_end,
                "sort_by": "exit_velocity",
                "limit": min(limit_value, 20)
            }
        )

    if len(results) < limit_value and ("count" in normalized_query or "how many" in normalized_query):
        snippet = f"Counts Statcast events (defaulting to home runs) from {default_start} to {default_end}."
        add_result(
            result_id=f"statcast_count:{year_context}",
            title=f"Count Statcast events {year_context}",
            snippet=snippet,
            tool="statcast_count",
            parameters={
                "start_date": default_start,
                "end_date": default_end,
                "result_type": "home_run"
            }
        )

    if not results:
        add_result(
            result_id="help",
            title="Available MLB tools",
            snippet="Try queries like 'Aaron Judge statcast', 'Yankees pitching stats 2025', or '2024 HR leaderboard'.",
            tool="help",
            parameters={}
        )

    return year_context, results[:limit_value]

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

def chunk_date_range(start_date: str, end_date: str, max_days: int = 30) -> List[tuple]:
    """Split date range into chunks to avoid API limits
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_days: Maximum days per chunk (default 30 for most queries)
    
    Returns:
        List of (start, end) date tuples
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    chunks = []
    
    # Calculate total days to determine optimal chunk size
    total_days = (end - start).days
    
    # For very large date ranges, use even bigger chunks
    if total_days > 365:
        max_days = min(60, max_days * 2)  # Double chunk size for multi-year queries
    
    while start <= end:
        chunk_end = min(start + timedelta(days=max_days - 1), end)
        chunks.append((start.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
        start = chunk_end + timedelta(days=1)
    
    return chunks

def get_cache_key(start_date: str, end_date: str, **kwargs) -> str:
    """Generate cache key for query"""
    key_data = f"{start_date}_{end_date}_{json.dumps(kwargs, sort_keys=True)}"
    return hashlib.md5(key_data.encode()).hexdigest()

def is_cache_valid(cache_entry: dict, ttl_seconds: int = 900) -> bool:
    """Check if cache entry is still valid (default 15 minute TTL)"""
    if not cache_entry:
        return False
    cached_time = datetime.fromisoformat(cache_entry['timestamp'])
    return (datetime.now() - cached_time).total_seconds() < ttl_seconds

def get_player_names_batch(player_ids: list) -> dict:
    """Get player names from IDs in batch with caching"""
    # First check cache
    result = {}
    missing_ids = []
    
    for pid in player_ids:
        if pid in player_name_cache:
            result[pid] = player_name_cache[pid]
        else:
            missing_ids.append(pid)
    
    # Look up missing IDs
    if missing_ids:
        try:
            pb = load_pybaseball()
            from pybaseball import playerid_reverse_lookup
            
            logger.info(f"Looking up {len(missing_ids)} player names...")
            # Look up all missing players at once
            player_data = playerid_reverse_lookup(missing_ids, key_type='mlbam')
            
            for _, player in player_data.iterrows():
                player_id = player['key_mlbam']
                full_name = f"{player['name_first']} {player['name_last']}"
                player_name_cache[player_id] = full_name
                result[player_id] = full_name
                
        except Exception as e:
            logger.debug(f"Could not look up players: {e}")
    
    # Add placeholder for any still missing
    for pid in player_ids:
        if pid not in result:
            result[pid] = f"Player {pid}"
    
    return result

def add_batter_names_to_data(data, batter_col='batter'):
    """Add batter_name column to statcast data efficiently
    
    Args:
        data: pandas DataFrame with statcast data
        batter_col: name of column containing batter IDs
        
    Returns:
        DataFrame with added batter_name column
    """
    if data.empty or batter_col not in data.columns:
        return data
    
    # Get unique batter IDs
    unique_batters = data[batter_col].dropna().unique()
    if len(unique_batters) == 0:
        return data
    
    # Batch lookup all names
    batter_names = get_player_names_batch(unique_batters.tolist())
    
    # Map names to data
    data['batter_name'] = data[batter_col].map(batter_names)

    return data

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def validate_date(date_str: str, param_name: str = "date") -> str:
    """Validate and return a YYYY-MM-DD date string.

    Raises ValueError with a user-friendly message on bad input.
    """
    if not date_str or not isinstance(date_str, str):
        raise ValueError(f"{param_name} is required and must be a string in YYYY-MM-DD format")
    date_str = date_str.strip()
    if not _DATE_RE.match(date_str):
        raise ValueError(f"{param_name} '{date_str}' is not in YYYY-MM-DD format")
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{param_name} '{date_str}' is not a valid calendar date")
    return date_str

def validate_year(year: Any) -> int:
    """Coerce *year* to int and ensure it is within the valid MLB range."""
    try:
        year = int(year)
    except (ValueError, TypeError):
        raise ValueError(f"year must be a number, got '{year}'")
    current_year = datetime.now().year
    if year < 1871 or year > current_year:
        raise ValueError(f"year must be between 1871 and {current_year}, got {year}")
    return year

def sanitize_player_name(name: str) -> str:
    """Basic sanitisation – strip whitespace and reject obviously bogus names."""
    if not name or not isinstance(name, str):
        raise ValueError("player name is required")
    name = name.strip()
    if len(name) < 2 or len(name) > 80:
        raise ValueError(f"player name '{name}' is too short or too long")
    # Allow letters, spaces, hyphens, apostrophes, periods
    if not re.match(r"^[A-Za-z .'\-]+$", name):
        raise ValueError(f"player name '{name}' contains invalid characters")
    return name

# ---------------------------------------------------------------------------
# Structured error responses
# ---------------------------------------------------------------------------

def _error_response(message: str, code: str = "server_error",
                    details: Optional[Dict[str, Any]] = None) -> str:
    """Return a consistently-structured JSON error string.

    *code* should be one of:
      - ``client_error``   – bad input, missing params, unknown player
      - ``external_error`` – upstream API failure, timeout
      - ``server_error``   – unexpected internal problem
    """
    payload: Dict[str, Any] = {"error": message, "code": code}
    if details:
        payload["details"] = details
    return json.dumps(payload)

# ---------------------------------------------------------------------------
# Player-ID lookup cache  (name -> MLBAM id)
# ---------------------------------------------------------------------------

# Maps  "first last" (lower) -> mlbam_id
_player_id_cache: Dict[str, int] = {}

def lookup_player_id(player_name: str) -> int:
    """Resolve *player_name* to an MLBAM ID, using a session-lifetime cache."""
    key = player_name.strip().lower()
    if key in _player_id_cache:
        return _player_id_cache[key]

    pb = load_pybaseball()
    from pybaseball import playerid_lookup

    parts = player_name.split()
    last_name = parts[-1]
    first_name = parts[0] if len(parts) > 1 else ""
    results = playerid_lookup(last_name, first_name)

    if results.empty and len(parts) >= 2:
        # Try flipped name order
        results = playerid_lookup(parts[0], " ".join(parts[1:]))

    if results.empty:
        raise LookupError(f"No player found matching '{player_name}'")

    mlbam_id = int(results.iloc[0]["key_mlbam"])
    _player_id_cache[key] = mlbam_id
    return mlbam_id

# ---------------------------------------------------------------------------
# Retry with exponential back-off
# ---------------------------------------------------------------------------

async def _fetch_with_retry(fn, *args, max_retries: int = 3,
                            base_delay: float = 1.0, **kwargs):
    """Run a blocking *fn* in a thread with retry + exponential backoff.

    This also wraps the call in ``asyncio.to_thread`` so it doesn't block the
    event loop.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                    f"{getattr(fn, '__name__', fn)}: {exc}  – retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed for "
                    f"{getattr(fn, '__name__', fn)}: {exc}"
                )
    raise last_exc  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

MAX_QUERY_CACHE_SIZE = 512

def _cache_put(key: str, data: str) -> None:
    """Store *data* in query_cache with a timestamp, pruning if needed."""
    query_cache[key] = {"data": data, "timestamp": datetime.now().isoformat()}
    # Prune oldest entries when cache exceeds limit
    if len(query_cache) > MAX_QUERY_CACHE_SIZE:
        oldest_keys = sorted(
            query_cache, key=lambda k: query_cache[k].get("timestamp", "")
        )[: len(query_cache) - MAX_QUERY_CACHE_SIZE]
        for k in oldest_keys:
            query_cache.pop(k, None)

@smithery.server()
def create_server():
    """Create and return a FastMCP server instance."""

    # Create the MCP server
    mcp = FastMCP("mlb-stats-mcp")

    original_create_init = mcp._mcp_server.create_initialization_options

    def create_init_with_search_capability(
        notification_options=None,
        experimental_capabilities: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        experimental_capabilities = experimental_capabilities or {}
        experimental_capabilities.setdefault(
            SEARCH_CAPABILITY_KEY,
            {
                "version": SEARCH_CAPABILITY_VERSION,
                "displayName": "MLB MCP Search",
                "supportsBatch": True,
            },
        )
        return original_create_init(
            notification_options=notification_options,
            experimental_capabilities=experimental_capabilities,
            **kwargs,
        )

    mcp._mcp_server.create_initialization_options = create_init_with_search_capability

    @mcp.tool()
    async def search(queries: Optional[List[Dict[str, Any]]] = None, limit: Optional[Any] = None) -> str:
        """
        Implements the MCP search connector contract. Accepts a list of SearchQuery objects and
        returns SearchResult objects that describe which MLB tools can satisfy each request.
        """
        if not queries:
            return json.dumps({"results": []}, indent=2)

        try:
            limit_value = max(1, min(int(limit), 20)) if limit is not None else 5
        except (ValueError, TypeError):
            limit_value = 5

        aggregated_query_results: List[Dict[str, Any]] = []

        for query_obj in queries:
            if not isinstance(query_obj, dict):
                continue
            query_text = (query_obj.get("query") or "").strip()
            if not query_text:
                continue

            year_context, candidates = generate_search_candidates(query_text, limit_value)
            structured_results: List[Dict[str, Any]] = []
            for candidate in candidates:
                candidate["query"] = query_text
                cache_payload = {
                    "tool": candidate.get("metadata", {}).get("tool"),
                    "parameters": candidate.get("metadata", {}).get("parameters"),
                    "title": candidate.get("title"),
                    "url": candidate.get("url"),
                    "snippet": candidate.get("snippet"),
                    "query": query_text,
                    "timestamp": datetime.now().isoformat()
                }
                _store_search_result_cache(candidate["id"], cache_payload)
                tool_name = candidate.get("metadata", {}).get("tool")
                tool_parameters = candidate.get("metadata", {}).get("parameters") or {}
                actions = []
                if tool_name:
                    actions.append(
                        {
                            "type": "tool",
                            "toolName": tool_name,
                            "arguments": tool_parameters,
                        }
                    )

                structured_results.append(
                    {
                        "id": candidate["id"],
                        "title": candidate.get("title"),
                        "snippet": candidate.get("snippet"),
                        "url": candidate.get("url"),
                        "score": candidate.get("score"),
                        "metadata": candidate.get("metadata"),
                        "actions": actions,
                        "year": year_context,
                    }
                )

            aggregated_query_results.append(
                {
                    "query": query_text,
                    "results": structured_results,
                }
            )

        response = {
            "results": aggregated_query_results
        }
        return json.dumps(response, indent=2, default=str)

    @mcp.tool()
    async def fetch(ids: Optional[List[str]] = None) -> str:
        """
        Executes the MCP search connector fetch contract. Accepts a list of result IDs and
        returns their fully-hydrated contents.
        """
        if not ids:
            return json.dumps({"results": []}, indent=2)

        normalized_ids = [result_id for result_id in ids if isinstance(result_id, str) and result_id.strip()]
        if not normalized_ids:
            return json.dumps({"results": []}, indent=2)

        results: List[Dict[str, Any]] = []

        for fetch_id in normalized_ids:
            cache_entry = search_result_cache.get(fetch_id)
            tool_name = (cache_entry or {}).get("tool")
            params = dict((cache_entry or {}).get("parameters") or {})

            if not tool_name:
                id_type = fetch_id.split(":")[0]
                tool_name = {
                    "player": "get_player_stats",
                    "team": "get_team_stats",
                    "leaderboard": "get_leaderboard",
                    "statcast_leaderboard": "statcast_leaderboard",
                    "statcast_count": "statcast_count",
                    "help": "help"
                }.get(id_type)

            if tool_name == "help":
                results.append({
                    "id": fetch_id,
                    "contents": [{
                        "type": "text",
                        "text": "Available tools: get_player_stats, get_team_stats, get_leaderboard, "
                                "statcast_leaderboard, statcast_count. Use the search tool to discover parameters."
                    }],
                    "metadata": {"tool": "help"}
                })
                continue

            if tool_name is None:
                results.append({
                    "id": fetch_id,
                    "error": {"message": f"Unrecognized fetch id '{fetch_id}'"}
                })
                continue

            id_parts = fetch_id.split(":")

            try:
                if tool_name == "get_player_stats":
                    name = params.get("name")
                    if not name:
                        if len(id_parts) >= 2 and id_parts[1].isdigit():
                            try:
                                pb = load_pybaseball()
                                from pybaseball import playerid_reverse_lookup
                                player_df = playerid_reverse_lookup([int(id_parts[1])], key_type='mlbam')
                                if not player_df.empty:
                                    row = player_df.iloc[0]
                                    name = f"{row.get('name_first', '').strip()} {row.get('name_last', '').strip()}".strip()
                                    params["name"] = name
                            except Exception as exc:
                                logger.debug(f"Reverse lookup failed for {fetch_id}: {exc}")
                    if not name:
                        raise ValueError("Player name is required to fetch stats.")
                    raw = await get_player_stats.fn(
                        name=name,
                        start_date=params.get("start_date"),
                        end_date=params.get("end_date")
                    )

                elif tool_name == "get_team_stats":
                    if not params and len(id_parts) >= 4:
                        params.update({
                            "team": id_parts[1],
                            "year": id_parts[2],
                            "stat_type": id_parts[3]
                        })
                    team = params.get("team")
                    year = params.get("year")
                    if not (team and year):
                        raise ValueError("Team and year are required for team stats.")
                    raw = await get_team_stats.fn(
                        team=team,
                        year=year,
                        stat_type=params.get("stat_type", "batting")
                    )

                elif tool_name == "get_leaderboard":
                    if not params and len(id_parts) >= 4:
                        params.update({
                            "stat": id_parts[1],
                            "leaderboard_type": id_parts[2],
                            "season": id_parts[3]
                        })
                    stat = params.get("stat", "HR")
                    season = params.get("season") or datetime.now().year
                    raw = await get_leaderboard.fn(
                        stat=stat,
                        season=season,
                        leaderboard_type=params.get("leaderboard_type", "batting"),
                        limit=params.get("limit", 10)
                    )

                elif tool_name == "statcast_leaderboard":
                    if not params and len(id_parts) >= 2:
                        year_val = id_parts[1]
                        params.update({
                            "start_date": f"{year_val}-03-01",
                            "end_date": f"{year_val}-11-15",
                            "sort_by": "exit_velocity",
                            "limit": 10
                        })
                    start_date = params.get("start_date")
                    end_date = params.get("end_date")
                    if not (start_date and end_date):
                        raise ValueError("start_date and end_date are required for statcast_leaderboard.")
                    raw = await statcast_leaderboard.fn(
                        start_date=start_date,
                        end_date=end_date,
                        result=params.get("result"),
                        min_ev=params.get("min_ev"),
                        min_pitch_velo=params.get("min_pitch_velo"),
                        sort_by=params.get("sort_by", "exit_velocity"),
                        limit=params.get("limit", 10),
                        order=params.get("order", "desc"),
                        group_by=params.get("group_by"),
                        pitch_type=params.get("pitch_type"),
                        player_name=params.get("player_name")
                    )

                elif tool_name == "statcast_count":
                    if not params and len(id_parts) >= 2:
                        year_val = id_parts[1]
                        params.update({
                            "start_date": f"{year_val}-03-01",
                            "end_date": f"{year_val}-11-15",
                            "result_type": "home_run"
                        })
                    start_date = params.get("start_date")
                    end_date = params.get("end_date")
                    if not (start_date and end_date):
                        raise ValueError("start_date and end_date are required for statcast_count.")
                    raw = await statcast_count.fn(
                        start_date=start_date,
                        end_date=end_date,
                        result_type=params.get("result_type", "home_run"),
                        min_distance=params.get("min_distance"),
                        min_exit_velocity=params.get("min_exit_velocity"),
                        max_distance=params.get("max_distance"),
                        max_exit_velocity=params.get("max_exit_velocity"),
                        player_name=params.get("player_name"),
                        pitch_type=params.get("pitch_type")
                    )

                else:
                    raise ValueError(f"Unsupported tool '{tool_name}'")

                content_text = raw if isinstance(raw, str) else json.dumps(raw, indent=2, default=str)
                fetch_result: Dict[str, Any] = {
                    "id": fetch_id,
                    "contents": [{
                        "type": "text",
                        "text": content_text
                    }]
                }
                if cache_entry and cache_entry.get("title"):
                    fetch_result["title"] = cache_entry["title"]
                if cache_entry and cache_entry.get("url"):
                    fetch_result["url"] = cache_entry["url"]
                metadata_out: Dict[str, Any] = {
                    "tool": tool_name,
                    "parameters": params
                }
                if cache_entry:
                    for key, value in cache_entry.items():
                        if key not in {"timestamp", "tool", "parameters"}:
                            metadata_out[key] = value
                fetch_result["metadata"] = metadata_out
                results.append(fetch_result)

            except Exception as exc:
                logger.error(f"Fetch failed for {fetch_id}: {exc}")
                results.append({
                    "id": fetch_id,
                    "error": {"message": str(exc)}
                })

        return json.dumps({"results": results}, indent=2, default=str)

    @mcp.tool(name="get_player_stats", description="Fetch player Statcast summary stats")
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
            name = sanitize_player_name(name)
            if start_date:
                start_date = validate_date(start_date, "start_date")
            if end_date:
                end_date = validate_date(end_date, "end_date")
        except ValueError as e:
            return _error_response(str(e), "client_error")

        try:
            player_id = await asyncio.to_thread(lookup_player_id, name)
        except LookupError as e:
            return _error_response(str(e), "client_error")

        try:
            from pybaseball import statcast_batter
            import pandas as pd

            if start_date and end_date:
                data = await _fetch_with_retry(statcast_batter, start_date, end_date, player_id)
            else:
                today = datetime.now()
                season_year = today.year if today.month >= 3 else today.year - 1
                season_start = f"{season_year}-03-01"
                data = await _fetch_with_retry(
                    statcast_batter, season_start, today.strftime('%Y-%m-%d'), player_id
                )

            if data.empty:
                return _error_response(f"No statcast data found for {name}", "client_error")

            hit_events = data['events'].isin(['single', 'double', 'triple', 'home_run'])
            stats = {
                "player": name,
                "games": int(data['game_date'].nunique()),
                "at_bats": len(data),
                "hits": int(hit_events.sum()),
                "home_runs": int((data['events'] == 'home_run').sum()),
                "avg_exit_velocity": round(float(data['launch_speed'].mean()), 2) if data['launch_speed'].notna().any() else None,
                "avg_launch_angle": round(float(data['launch_angle'].mean()), 2) if data['launch_angle'].notna().any() else None,
                "barrel_rate": round(float((data['barrel'] == 1).sum() / len(data) * 100), 2) if 'barrel' in data.columns else None
            }

            return json.dumps(stats, indent=2)

        except Exception as e:
            logger.error(f"Error getting player stats: {str(e)}")
            return _error_response(str(e), "external_error")
    
    @mcp.tool(name="get_team_stats", description="Fetch team batting or pitching totals for a season")
    async def get_team_stats(team: str, year: Any, stat_type: str = "batting") -> str:
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
            year = validate_year(year)
        except ValueError as e:
            return _error_response(str(e), "client_error")

        stat_type = (stat_type or "batting").lower()
        if stat_type not in ("batting", "pitching"):
            return _error_response(
                f"Invalid stat_type: {stat_type}. Use 'batting' or 'pitching'",
                "client_error",
            )

        try:
            load_pybaseball()
            from pybaseball import team_batting, team_pitching
            import pandas as pd

            logger.info(f"Fetching {stat_type} stats for {year}")
            fetch_fn = team_batting if stat_type == "batting" else team_pitching
            data = await _fetch_with_retry(fetch_fn, year)

            team_abbr = resolve_team_alias(team)
            
            # Try exact match first
            team_data = data[data['Team'] == team_abbr]
            
            # If not found, try contains search
            if team_data.empty:
                team_data = data[data['Team'].str.contains(team_abbr, case=False, na=False)]
            
            # If still not found, try original team name
            if team_data.empty:
                team_data = data[data['Team'].str.contains(team, case=False, na=False)]
            
            if team_data.empty:
                available_teams = data['Team'].unique().tolist()
                return _error_response(
                    f"No data found for team '{team}' in {year}",
                    "client_error",
                    {"available_teams": sorted(available_teams)},
                )

            result = team_data.iloc[0].to_dict()

            import numpy as np
            for key, value in result.items():
                if isinstance(value, float) and np.isnan(value):
                    result[key] = None

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting team stats: {str(e)}")
            return _error_response(str(e), "external_error")
    
    @mcp.tool(name="get_leaderboard", description="Get MLB leaderboard for a stat and season")
    async def get_leaderboard(stat: str, season: Any, leaderboard_type: str = "batting", limit: Any = 10) -> str:
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
            season = validate_year(season)
        except ValueError as e:
            return _error_response(str(e), "client_error")

        try:
            limit = max(1, min(int(limit), 50))
        except (ValueError, TypeError):
            limit = 10

        leaderboard_type = (leaderboard_type or "batting").lower()
        if leaderboard_type not in ("batting", "pitching"):
            return _error_response(
                f"Invalid leaderboard_type: {leaderboard_type}. Use 'batting' or 'pitching'",
                "client_error",
            )

        try:
            load_pybaseball()
            from pybaseball import batting_stats, pitching_stats

            fetch_fn = batting_stats if leaderboard_type == "batting" else pitching_stats
            data = await _fetch_with_retry(fetch_fn, season)

            if stat not in data.columns:
                return _error_response(
                    f"Stat '{stat}' not found",
                    "client_error",
                    {"available_stats": list(data.columns)},
                )

            sorted_data = data.sort_values(by=stat, ascending=False).head(limit)

            leaderboard = []
            for rank, (idx, row) in enumerate(sorted_data.iterrows(), 1):
                entry = {
                    "rank": rank,
                    "player": row.get('Name', 'Unknown'),
                    "team": row.get('Team', 'Unknown'),
                    stat: row[stat]
                }
                leaderboard.append(entry)

            return json.dumps({
                "stat": stat,
                "season": season,
                "type": leaderboard_type,
                "leaderboard": leaderboard
            }, indent=2)

        except Exception as e:
            logger.error(f"Error getting leaderboard: {str(e)}")
            return _error_response(str(e), "external_error")
    
    @mcp.tool()
    async def team_season_stats(year: Any, stat: str = "exit_velocity", min_result_type: Optional[str] = None) -> str:
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
            year = validate_year(year)
        except ValueError as e:
            return _error_response(str(e), "client_error")

        VALID_STATS = {'exit_velocity', 'distance', 'launch_angle', 'barrel_rate', 'hard_hit_rate', 'sweet_spot_rate'}
        if stat not in VALID_STATS:
            return _error_response(
                f"Invalid stat: {stat}. Use one of: {', '.join(sorted(VALID_STATS))}",
                "client_error",
            )

        try:
            cache_key = f"team_season_{year}_{stat}_{min_result_type}"
            if is_cache_valid(query_cache.get(cache_key), ttl_seconds=86400):
                logger.info(f"Using cached team season data for {cache_key}")
                return query_cache[cache_key]['data']

            load_pybaseball()
            from pybaseball import statcast
            import pandas as pd
            import numpy as np

            if year == datetime.now().year:
                start_date = f"{year}-04-01"
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                start_date = f"{year}-04-01"
                end_date = f"{year}-10-31"

            logger.info(f"Fetching team season stats for {year}, stat={stat}")

            # Use chunked sampling: fetch 2-day windows every 14 days for representative data
            # This reduces API calls by ~3.5x vs fetching individual days every 7 days
            all_data = []
            current = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            sample_count = 0

            while current <= end_dt:
                sample_end = min(current + timedelta(days=1), end_dt)
                try:
                    data = await _fetch_with_retry(
                        statcast,
                        start_dt=current.strftime('%Y-%m-%d'),
                        end_dt=sample_end.strftime('%Y-%m-%d'),
                        max_retries=2,
                    )
                    if not data.empty:
                        all_data.append(data)
                    sample_count += 1
                except Exception as e:
                    logger.debug(f"Error fetching data for {current.strftime('%Y-%m-%d')}: {str(e)}")
                current += timedelta(days=14)

            logger.info(f"Fetched {sample_count} sample windows for team season stats")

            if not all_data:
                return _error_response(f"No data available for {year} season", "external_error")

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
                "sample_size": f"{sample_count} sample windows",
                "total_events": len(data),
                "leaderboard": leaderboard
            }, indent=2, default=str)
            
            _cache_put(cache_key, response)
            return response

        except Exception as e:
            logger.error(f"Error getting team season stats: {str(e)}")
            return _error_response(str(e), "external_error")

    @mcp.tool()
    async def team_pitching_stats(year: Any, stat: str = "velocity", pitch_type: Optional[str] = None) -> str:
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
            year = validate_year(year)
        except ValueError as e:
            return _error_response(str(e), "client_error")

        VALID_PITCH_STATS = {'velocity', 'spin_rate', 'movement', 'whiff_rate', 'chase_rate', 'zone_rate', 'ground_ball_rate', 'xera'}
        if stat not in VALID_PITCH_STATS:
            return _error_response(
                f"Invalid stat: {stat}. Use one of: {', '.join(sorted(VALID_PITCH_STATS))}",
                "client_error",
            )

        try:
            cache_key = f"team_pitching_{year}_{stat}_{pitch_type}"
            if is_cache_valid(query_cache.get(cache_key), ttl_seconds=86400):
                logger.info(f"Using cached team pitching data for {cache_key}")
                return query_cache[cache_key]['data']

            load_pybaseball()
            from pybaseball import statcast
            import pandas as pd
            import numpy as np

            if year == datetime.now().year:
                start_date = f"{year}-04-01"
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                start_date = f"{year}-04-01"
                end_date = f"{year}-10-31"

            logger.info(f"Fetching team pitching stats for {year}, stat={stat}")

            all_data = []
            current = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            sample_count = 0

            while current <= end_dt:
                sample_end = min(current + timedelta(days=1), end_dt)
                try:
                    data = await _fetch_with_retry(
                        statcast,
                        start_dt=current.strftime('%Y-%m-%d'),
                        end_dt=sample_end.strftime('%Y-%m-%d'),
                        max_retries=2,
                    )
                    if not data.empty:
                        all_data.append(data)
                    sample_count += 1
                except Exception as e:
                    logger.debug(f"Error fetching data for {current.strftime('%Y-%m-%d')}: {str(e)}")
                current += timedelta(days=14)

            logger.info(f"Fetched {sample_count} sample windows for team pitching stats")

            if not all_data:
                return _error_response(f"No data available for {year} season", "external_error")

            data = pd.concat(all_data, ignore_index=True)

            if pitch_type:
                data = data[data['pitch_type'] == pitch_type]
                if data.empty:
                    return _error_response(f"No data found for pitch type {pitch_type}", "client_error")
            
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
                "sample_size": f"{sample_count} sample windows",
                "total_pitches": len(data),
                "leaderboard": leaderboard
            }, indent=2, default=str)
            
            _cache_put(cache_key, response)
            return response

        except Exception as e:
            logger.error(f"Error getting team pitching stats: {str(e)}")
            return _error_response(str(e), "external_error")

    @mcp.tool()
    async def statcast_count(start_date: str, end_date: str, result_type: str = "home_run",
                            min_distance: Optional[Any] = None, min_exit_velocity: Optional[Any] = None,
                            max_distance: Optional[Any] = None, max_exit_velocity: Optional[Any] = None,
                            player_name: Optional[str] = None, pitch_type: Optional[str] = None) -> str:
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
            player_name: Filter by batter name (e.g., 'Aaron Judge', 'Mike Trout')
            pitch_type: Filter by pitch type - 'FF' (4-seam), 'SL' (slider), 'CH' (changeup), 'CU' (curveball), 'SI' (sinker), 'FC' (cutter), 'FS' (splitter)
    
        Returns:
            JSON with count, breakdown by year, and top examples
        """
        try:
            start_date = validate_date(start_date, "start_date")
            end_date = validate_date(end_date, "end_date")
        except ValueError as e:
            return _error_response(str(e), "client_error")

        try:
            if min_distance is not None:
                min_distance = float(min_distance)
            if max_distance is not None:
                max_distance = float(max_distance)
            if min_exit_velocity is not None:
                min_exit_velocity = float(min_exit_velocity)
            if max_exit_velocity is not None:
                max_exit_velocity = float(max_exit_velocity)
        except (ValueError, TypeError) as e:
            return _error_response(f"Invalid numeric filter: {e}", "client_error")

        try:
            player_id = None
            if player_name:
                try:
                    player_name = sanitize_player_name(player_name)
                    player_id = await asyncio.to_thread(lookup_player_id, player_name)
                except (ValueError, LookupError) as e:
                    return _error_response(str(e), "client_error")

            cache_key = get_cache_key(start_date, end_date, result_type=result_type,
                                      min_distance=min_distance, min_exit_velocity=min_exit_velocity,
                                      max_distance=max_distance, max_exit_velocity=max_exit_velocity,
                                      player_id=player_id, pitch_type=pitch_type)
            if is_cache_valid(query_cache.get(cache_key), ttl_seconds=86400):
                    logger.info(f"Using cached count data for {cache_key}")
                    return query_cache[cache_key]['data']
            
            pb = load_pybaseball()
            from pybaseball import statcast
            import pandas as pd
            import numpy as np
            
            logger.info(f"Counting {result_type} from {start_date} to {end_date}")
            
            # Use optimized chunking for all queries
            date_chunks = chunk_date_range(start_date, end_date, max_days=45)  # Larger chunks for counting
            all_data = []
            total_count = 0
            yearly_counts = {}
            
            logger.info(f"Query will fetch data in {len(date_chunks)} chunks")
            
            for i, (chunk_start, chunk_end) in enumerate(date_chunks, 1):
                try:
                    logger.info(f"Fetching chunk {i}/{len(date_chunks)}: {chunk_start} to {chunk_end}")
                    chunk_data = await _fetch_with_retry(
                        statcast, start_dt=chunk_start, end_dt=chunk_end, max_retries=2,
                    )

                    if not chunk_data.empty:
                        # Apply filters immediately to reduce memory
                        if result_type:
                            if result_type == 'batted_ball':
                                chunk_data = chunk_data[chunk_data['type'] == 'X']
                            elif result_type == 'hit':
                                chunk_data = chunk_data[chunk_data['events'].isin(['single', 'double', 'triple', 'home_run'])]
                            else:
                                chunk_data = chunk_data[chunk_data['events'] == result_type]
    
                        # Apply player filter
                        if player_id:
                            chunk_data = chunk_data[chunk_data['batter'] == player_id]
    
                        # Apply pitch type filter
                        if pitch_type:
                            chunk_data = chunk_data[chunk_data['pitch_type'] == pitch_type]
    
                        # Apply distance and exit velocity filters
                        if min_distance:
                            chunk_data = chunk_data[chunk_data['hit_distance_sc'] >= min_distance]
                        if max_distance:
                            chunk_data = chunk_data[chunk_data['hit_distance_sc'] <= max_distance]
                        if min_exit_velocity:
                            chunk_data = chunk_data[chunk_data['launch_speed'] >= min_exit_velocity]
                        if max_exit_velocity:
                            chunk_data = chunk_data[chunk_data['launch_speed'] <= max_exit_velocity]
                        
                        if not chunk_data.empty:
                            # Count by year
                            chunk_data['year'] = pd.to_datetime(chunk_data['game_date']).dt.year
                            year_counts = chunk_data['year'].value_counts().to_dict()
                            
                            for year, count in year_counts.items():
                                yearly_counts[year] = yearly_counts.get(year, 0) + count
                                total_count += count
                            
                            all_data.append(chunk_data)
                            logger.info(f"  Found {len(chunk_data)} matching events in this chunk")
                            
                except Exception as e:
                    logger.error(f"Error fetching chunk {chunk_start} to {chunk_end}: {str(e)}")
                    continue
            
            # Combine all data for examples
            if all_data:
                data = pd.concat(all_data, ignore_index=True)
            else:
                data = pd.DataFrame()
            
            if data.empty:
                return json.dumps({
                    "query": f"Count of {result_type or 'all events'} from {start_date} to {end_date}",
                    "total_count": 0,
                    "yearly_breakdown": {},
                    "filters": {
                        "result_type": result_type,
                        "player_name": player_name,
                        "pitch_type": pitch_type,
                        "min_distance": min_distance,
                        "max_distance": max_distance,
                        "min_exit_velocity": min_exit_velocity,
                        "max_exit_velocity": max_exit_velocity
                    }
                })
            
            # Add batter names for examples
            data = add_batter_names_to_data(data)
            
            # Get top examples
            examples = []
            if not data.empty:
                # Sort by distance for home runs, exit velocity for others
                sort_col = 'hit_distance_sc' if result_type == 'home_run' and 'hit_distance_sc' in data.columns else 'launch_speed'
                top_data = data.nlargest(5, sort_col)
                
                for _, row in top_data.iterrows():
                    example = {
                        'player': str(row.get('batter_name', f"Player {row.get('batter', 'Unknown')}")),
                        'pitcher': str(row.get('player_name', 'Unknown')),  # player_name is the pitcher
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
            query_desc = f"Count of {result_type or 'all events'}"
            if player_name:
                query_desc += f" for {player_name}"
            if pitch_type:
                query_desc += f" on {pitch_type} pitches"
            query_desc += f" from {start_date} to {end_date}"
    
            response = {
                'query': query_desc,
                'start_date': start_date,
                'end_date': end_date,
                'filters': {
                    'result_type': result_type,
                    'player_name': player_name,
                    'pitch_type': pitch_type,
                    'min_distance': min_distance,
                    'max_distance': max_distance,
                    'min_exit_velocity': min_exit_velocity,
                    'max_exit_velocity': max_exit_velocity
                },
                'total_count': total_count,
                'yearly_breakdown': yearly_counts,
                'top_examples': examples
            }
            
            response_json = json.dumps(response, indent=2, default=str)
            
            _cache_put(cache_key, response_json)
            return response_json

        except Exception as e:
            logger.error(f"Error in statcast_count: {str(e)}")
            return _error_response(str(e), "external_error")
    
    @mcp.tool()
    async def top_home_runs(year_start: Any = 2023, year_end: Optional[Any] = None, 
                            limit: Any = 5, min_exit_velocity: Optional[Any] = None) -> str:
        """
        Get top home runs by exit velocity for a given year range. Optimized for performance.
        
        Args:
            year_start: Starting year (default: 2023)
            year_end: Ending year (default: current year)
            limit: Number of results to return (default: 5)
            min_exit_velocity: Minimum exit velocity filter in mph (optional)
        
        Returns:
            JSON string of top home runs
        """
        try:
            year_start = validate_year(year_start)
            year_end = validate_year(year_end) if year_end is not None else datetime.now().year
        except ValueError as e:
            return _error_response(str(e), "client_error")

        try:
            limit = max(1, min(int(limit), 50))
        except (ValueError, TypeError):
            limit = 5

        try:
            if min_exit_velocity is not None:
                min_exit_velocity = float(min_exit_velocity)
        except (ValueError, TypeError) as e:
            return _error_response(f"Invalid min_exit_velocity: {e}", "client_error")

        try:
            cache_key = f"top_hr_{year_start}_{year_end}_{limit}_{min_exit_velocity}"
            if cache_key in query_cache and is_cache_valid(query_cache[cache_key]):
                logger.info(f"Using cached data for top home runs query")
                return query_cache[cache_key]['data']

            load_pybaseball()
            from pybaseball import statcast
            import pandas as pd
            import numpy as np
            
            # Maintain a running list of top home runs
            top_hrs = []
            
            logger.info(f"Fetching top {limit} home runs from {year_start} to {year_end}")
            
            # Query month by month during baseball season only
            for year in range(year_start, year_end + 1):
                # Baseball season is typically March to October
                for month in range(3, 11):  # March to October
                    try:
                        # Calculate month date range
                        start_date = f"{year}-{month:02d}-01"
                        if month == 10:  # October
                            end_date = f"{year}-{month:02d}-31"
                        else:
                            # Last day of month
                            if month in [4, 6, 9]:  # April, June, September
                                end_date = f"{year}-{month:02d}-30"
                            else:
                                end_date = f"{year}-{month:02d}-31"
                        
                        logger.info(f"Checking {year} {calendar.month_name[month]}")
                        
                        # Query only home runs for this month
                        month_data = await _fetch_with_retry(
                            statcast, start_dt=start_date, end_dt=end_date, max_retries=2,
                        )
                        
                        if not month_data.empty:
                            # Filter home runs
                            hrs = month_data[month_data['events'] == 'home_run']
                            
                            # Apply exit velocity filter if specified
                            if min_exit_velocity:
                                hrs = hrs[hrs['launch_speed'] >= min_exit_velocity]
                            
                            if not hrs.empty:
                                # Filter out rows with missing exit velocity or batter
                                hrs = hrs.dropna(subset=['launch_speed', 'batter'])

                                if not hrs.empty:
                                    # Only keep candidates that could make the top N
                                    min_threshold = top_hrs[-1]['exit_velocity'] if len(top_hrs) >= limit else 0
                                    hrs = hrs[hrs['launch_speed'] > min_threshold]

                                if not hrs.empty:
                                    # Use itertuples for faster iteration (5-10x faster than iterrows)
                                    for hr in hrs.itertuples():
                                        try:
                                            top_hrs.append({
                                                'exit_velocity': float(hr.launch_speed),
                                                'batter_id': int(hr.batter),
                                                'player': f"Batter {int(hr.batter)}",
                                                'pitcher': str(getattr(hr, 'player_name', 'Unknown')),
                                                'date': str(getattr(hr, 'game_date', 'Unknown')),
                                                'distance': float(hr.hit_distance_sc) if pd.notna(hr.hit_distance_sc) else 0,
                                                'launch_angle': float(hr.launch_angle) if pd.notna(hr.launch_angle) else 0,
                                                'pitch_velocity': float(hr.release_speed) if pd.notna(hr.release_speed) else 0,
                                                'pitch_type': str(getattr(hr, 'pitch_type', 'Unknown')),
                                                'team': str(getattr(hr, 'batting_team', getattr(hr, 'home_team', 'Unknown'))),
                                                'description': str(getattr(hr, 'des', 'No description')),
                                                'game_pk': str(getattr(hr, 'game_pk', ''))
                                            })
                                        except (ValueError, TypeError) as e:
                                            logger.debug(f"Skipping row due to conversion error: {e}")
                                            continue

                                    # Keep only top N
                                    top_hrs = sorted(top_hrs, key=lambda x: x['exit_velocity'], reverse=True)[:limit]
                                    logger.info(f"  Found {len(hrs)} home runs, current top EV: {top_hrs[0]['exit_velocity']:.1f} mph")
                        
                    except Exception as e:
                        logger.error(f"Error fetching {year}-{month:02d}: {str(e)}")
                        continue
            
            # Look up all batter names at once
            logger.info("Looking up player names...")
            batter_ids = [hr['batter_id'] for hr in top_hrs if 'batter_id' in hr]
            if batter_ids:
                name_map = get_player_names_batch(batter_ids)
                for hr in top_hrs:
                    if 'batter_id' in hr:
                        hr['player'] = name_map.get(hr['batter_id'], hr['player'])
            
            # Format final results
            result = {
                "query": f"Top {limit} hardest hit home runs from {year_start} to {year_end}",
                "min_exit_velocity_filter": min_exit_velocity,
                "results": []
            }
            
            for i, hr in enumerate(top_hrs, 1):
                # Add video links
                video_info = {}
                if hr['game_pk']:
                    video_info['game_highlights_url'] = f"https://www.mlb.com/gameday/{hr['game_pk']}/video"
                    player_name = hr['player'].replace(' ', '+')
                    video_info['film_room_search'] = f"https://www.mlb.com/video/search?q={player_name}+{hr['date']}"
                
                result["results"].append({
                    "rank": i,
                    "player": hr['player'],
                    "pitcher": hr.get('pitcher', 'Unknown'),
                    "date": hr['date'],
                    "exit_velocity": hr['exit_velocity'],
                    "distance": hr['distance'],
                    "launch_angle": hr['launch_angle'],
                    "pitch": f"{hr['pitch_velocity']} mph {hr['pitch_type']}",
                    "team": hr['team'],
                    "description": hr['description'],
                    "video_links": video_info if video_info else None
                })
            
            response = json.dumps(result, indent=2)
            
            _cache_put(cache_key, response)
            return response

        except Exception as e:
            logger.error(f"Error getting top home runs: {str(e)}")
            return _error_response(str(e), "external_error")
    
    @mcp.tool()
    async def statcast_leaderboard(start_date: str, end_date: str, result: Optional[str] = None,
                                 min_ev: Optional[float] = None, min_pitch_velo: Optional[float] = None,
                                 sort_by: str = "exit_velocity", limit: int = 10, order: str = "desc",
                                 group_by: Optional[str] = None, pitch_type: Optional[str] = None,
                                 player_name: Optional[str] = None) -> str:
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
            pitch_type: Filter by pitch type - 'FF' (4-seam), 'SL' (slider), 'CH' (changeup), 'CU' (curveball), 'SI' (sinker), 'FC' (cutter), 'FS' (splitter) (optional)
            player_name: Filter by batter name (e.g., 'Aaron Judge', 'Mike Trout') (optional)
    
        Returns:
            JSON string of statcast leaderboard
        """
        try:
            start_date = validate_date(start_date, "start_date")
            end_date = validate_date(end_date, "end_date")
        except ValueError as e:
            return _error_response(str(e), "client_error")

        try:
            player_id = None
            if player_name:
                try:
                    player_name = sanitize_player_name(player_name)
                    player_id = await asyncio.to_thread(lookup_player_id, player_name)
                except (ValueError, LookupError) as e:
                    return _error_response(str(e), "client_error")

            cache_key = get_cache_key(start_date, end_date, result=result, min_ev=min_ev,
                                     min_pitch_velo=min_pitch_velo, sort_by=sort_by,
                                     limit=limit, order=order, group_by=group_by,
                                     pitch_type=pitch_type, player_id=player_id)

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
            
            logger.info(f"Query will fetch data in {len(date_chunks)} chunks")
            
            for i, (chunk_start, chunk_end) in enumerate(date_chunks, 1):
                try:
                    logger.info(f"Fetching chunk {i}/{len(date_chunks)}: {chunk_start} to {chunk_end}")
                    chunk_data = await _fetch_with_retry(
                        statcast, start_dt=chunk_start, end_dt=chunk_end, max_retries=2,
                    )
                    if not chunk_data.empty:
                        # Apply all possible filters early to reduce memory usage
                        if result:
                            chunk_data = chunk_data[chunk_data['events'] == result]
                        if min_ev:
                            chunk_data = chunk_data[chunk_data['launch_speed'] >= min_ev]
                        if min_pitch_velo:
                            chunk_data = chunk_data[chunk_data['release_speed'] >= min_pitch_velo]
                        if player_id:
                            chunk_data = chunk_data[chunk_data['batter'] == player_id]
                        if not chunk_data.empty:
                            all_data.append(chunk_data)
                            logger.info(f"  Found {len(chunk_data)} matching events in this chunk")
                except Exception as e:
                    logger.error(f"Error fetching chunk {chunk_start} to {chunk_end}: {str(e)}")
                    continue
            
            if not all_data:
                return _error_response(f"No statcast data found for date range {start_date} to {end_date}", "client_error")
            
            # Combine all chunks
            data = pd.concat(all_data, ignore_index=True) if len(all_data) > 1 else all_data[0]
            
            # Apply pitch type filter (complex logic not suitable for chunk loop)
            if pitch_type:
                pitch_filter = None
                if isinstance(pitch_type, str):
                    pitch_type_upper = pitch_type.upper()
                    if pitch_type_upper == 'FASTBALL':
                        pitch_filter = ['FF', 'FA', 'FT', 'SI', 'FC']
                    elif ',' in pitch_type_upper:
                        pitch_filter = [code.strip() for code in pitch_type_upper.split(',') if code.strip()]
                    else:
                        pitch_filter = [pitch_type_upper]
                elif isinstance(pitch_type, (list, tuple, set)):
                    pitch_filter = [str(code).upper() for code in pitch_type]

                if pitch_filter:
                    data = data[data['pitch_type'].isin(pitch_filter)]
                else:
                    data = data[data['pitch_type'] == pitch_type]

                if data.empty:
                    return _error_response(f"No pitches matching pitch filter {pitch_type} found in the specified criteria", "client_error")

            if data.empty:
                return _error_response("No data found matching the specified criteria", "client_error")
            
            # Add batter names after all filtering to minimize lookups
            data = add_batter_names_to_data(data)

            # Determine sort column
            sort_column_map = {
                'exit_velocity': 'launch_speed',
                'distance': 'hit_distance_sc',
                'launch_angle': 'launch_angle',
                'pitch_velocity': 'release_speed',
                'spin_rate': 'release_spin_rate',
                'xba': 'estimated_ba_using_speedangle',
                'xwoba': 'estimated_woba_using_speedangle',
                'barrel': 'barrel',
                'count': None,
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
                # Remove rows with null values when applicable
                if sort_column:
                    data = data.dropna(subset=[sort_column])
                    if data.empty:
                        return _error_response(f"No data available for sorting by {sort_by}", "client_error")
                
                # Group by team and calculate statistics
                team_stats = data.groupby('batting_team').agg({
                    sort_column: ['mean', 'max', 'count'],
                    'launch_speed': 'mean',
                    'hit_distance_sc': 'mean',
                    'release_speed': 'mean',
                    'release_spin_rate': 'mean',
                    'estimated_ba_using_speedangle': 'mean',
                    'estimated_woba_using_speedangle': 'mean',
                }).round(2)

                # Calculate barrel count separately to avoid lambda naming issues
                if 'barrel' in data.columns:
                    barrel_counts = data.groupby('batting_team')['barrel'].apply(lambda x: (x == 1).sum())
                    team_stats['barrel_count'] = barrel_counts
                else:
                    team_stats['barrel_count'] = 0

                # Flatten column names
                team_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in team_stats.columns.values]
                
                # Sort by the main metric
                sort_col_for_team = 'event_count'
                if sort_by != 'count':
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
                        sort_for_highlight = sort_column or 'launch_speed'
                        top_play = team_data.nlargest(1, sort_for_highlight).iloc[0]
                        
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
                        "count": int(row.get(f"{sort_column}_count", row.get('event_count', 0))),
                        f"{sort_by}_avg": float(row.get(f"{sort_column}_mean", 0)) if sort_by != 'count' else None,
                        f"{sort_by}_max": float(row.get(f"{sort_column}_max", 0)) if sort_by != 'count' else None,
                        "avg_exit_velocity": float(row.get('launch_speed_mean', 0)) if 'launch_speed_mean' in row else None,
                        "avg_distance": float(row.get('hit_distance_sc_mean', 0)) if 'hit_distance_sc_mean' in row else None,
                        "avg_pitch_velocity": float(row.get('release_speed_mean', 0)) if 'release_speed_mean' in row else None,
                        "avg_spin_rate": float(row.get('release_spin_rate_mean', 0)) if 'release_spin_rate_mean' in row else None,
                        "avg_xba": float(row.get('estimated_ba_using_speedangle_mean', 0)) if 'estimated_ba_using_speedangle_mean' in row else None,
                        "avg_xwoba": float(row.get('estimated_woba_using_speedangle_mean', 0)) if 'estimated_woba_using_speedangle_mean' in row else None,
                        "barrel_count": int(row.get('barrel_count', 0)),
                        "top_play_video": top_play_video if top_play_video else None
                    }
                    leaderboard.append(entry)
                
                # Create response for team grouping
                response = json.dumps({
                    "start_date": start_date,
                    "end_date": end_date,
                    "filter": {
                        "result": result,
                        "min_ev": min_ev,
                        "min_pitch_velo": min_pitch_velo,
                        "pitch_type": pitch_type,
                        "player_name": player_name
                    },
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

            # Handle player grouping if requested
            if group_by == 'player':
                if sort_column:
                    data = data.dropna(subset=[sort_column])
                    if data.empty:
                        return _error_response(f"No data available for sorting by {sort_by}", "client_error")

                data['player_display'] = data['batter_name'].where(data['batter_name'].notna(), data['batter'].apply(lambda x: f"Player {x}"))
                data['player_id'] = data['batter']

                # Aggregate by player
                agg_config = {
                    'launch_speed': ['mean', 'max'],
                    'hit_distance_sc': 'mean',
                    'release_speed': 'mean',
                    'release_spin_rate': 'mean',
                    'estimated_ba_using_speedangle': 'mean',
                    'estimated_woba_using_speedangle': 'mean',
                    'events': 'count'
                }
                if sort_column and sort_column not in agg_config:
                    agg_config[sort_column] = ['mean', 'max']

                player_stats = data.groupby(['player_id', 'player_display']).agg(agg_config).round(2)

                player_stats.columns = ['_'.join(col).strip() for col in player_stats.columns.values]
                player_stats = player_stats.rename(columns={
                    'events_count': 'event_count',
                    'launch_speed_mean': 'avg_exit_velocity',
                    'launch_speed_max': 'max_exit_velocity',
                    'hit_distance_sc_mean': 'avg_distance',
                    'release_speed_mean': 'avg_pitch_velocity',
                    'release_speed_max': 'max_pitch_velocity',
                    'release_spin_rate_mean': 'avg_spin_rate',
                    'estimated_ba_using_speedangle_mean': 'avg_xba',
                    'estimated_woba_using_speedangle_mean': 'avg_xwoba'
                })
                if 'event_count' not in player_stats.columns and 'events_count' in player_stats.columns:
                    player_stats['event_count'] = player_stats['events_count']

                # Determine sorting column for players
                if sort_by == 'count':
                    player_stats = player_stats.sort_values(by='event_count', ascending=(order.lower() == 'asc'))
                elif sort_column:
                    sort_col_mean = f"{sort_column}_mean"
                    sort_col_max = f"{sort_column}_max"
                    sort_col_for_player = sort_col_max if sort_by in ['distance', 'exit_velocity', 'pitch_velocity'] else sort_col_mean
                    player_stats = player_stats.sort_values(by=sort_col_for_player, ascending=(order.lower() == 'asc'))
                else:
                    player_stats = player_stats.sort_values(by='event_count', ascending=False)

                player_stats = player_stats.head(limit)

                leaderboard = []
                for idx, ((player_id, player_name), row) in enumerate(player_stats.iterrows()):
                    player_data = data[data['player_id'] == player_id]
                    team_counts = player_data['batting_team'].value_counts()
                    primary_team = team_counts.index[0] if not team_counts.empty else 'Unknown'

                    # Highlight play for player
                    highlight_sort_column = sort_column or 'launch_speed'
                    highlight_play = player_data.nlargest(1, highlight_sort_column).iloc[0]
                    highlight = {
                        'date': str(highlight_play.get('game_date', 'Unknown')),
                        'pitcher': str(highlight_play.get('player_name', 'Unknown')),
                        'pitch_velocity': float(highlight_play.get('release_speed')) if pd.notna(highlight_play.get('release_speed')) else None,
                        'exit_velocity': float(highlight_play.get('launch_speed')) if pd.notna(highlight_play.get('launch_speed')) else None,
                        'distance': float(highlight_play.get('hit_distance_sc')) if pd.notna(highlight_play.get('hit_distance_sc')) else None,
                        'pitch_type': str(highlight_play.get('pitch_type', 'Unknown')),
                        'game_pk': str(highlight_play.get('game_pk')) if highlight_play.get('game_pk') else None,
                    }
                    if highlight.get('game_pk'):
                        player_slug = player_name.replace(' ', '+')
                        game_date = highlight['date']
                        highlight['video_links'] = {
                            'game_highlights_url': f"https://www.mlb.com/gameday/{highlight['game_pk']}/video",
                            'film_room_search': f"https://www.mlb.com/video/search?q={player_slug}+{game_date}" if player_slug and game_date else None
                        }

                    leaderboard.append({
                        "rank": idx + 1,
                        "player": player_name,
                        "player_id": int(player_id) if pd.notna(player_id) else None,
                        "team": primary_team,
                        "count": int(row.get('event_count', 0)),
                        "avg_exit_velocity": float(row.get('avg_exit_velocity', 0)) if 'avg_exit_velocity' in row else None,
                        "max_exit_velocity": float(row.get('max_exit_velocity', 0)) if 'max_exit_velocity' in row else None,
                        "avg_distance": float(row.get('avg_distance', 0)) if 'avg_distance' in row else None,
                        "avg_pitch_velocity": float(row.get('avg_pitch_velocity', 0)) if 'avg_pitch_velocity' in row else None,
                        "avg_spin_rate": float(row.get('avg_spin_rate', 0)) if 'avg_spin_rate' in row else None,
                        "avg_xba": float(row.get('avg_xba', 0)) if 'avg_xba' in row else None,
                        "avg_xwoba": float(row.get('avg_xwoba', 0)) if 'avg_xwoba' in row else None,
                        "highlight": highlight,
                    })

                response = json.dumps({
                    "start_date": start_date,
                    "end_date": end_date,
                    "filter": {
                        "result": result,
                        "min_ev": min_ev,
                        "min_pitch_velo": min_pitch_velo,
                        "pitch_type": pitch_type,
                        "player_name": player_name,
                    },
                    "grouping": "player",
                    "sorted_by": sort_by,
                    "leaderboard": leaderboard,
                }, indent=2, default=str)

                _cache_put(cache_key, response)
                return response
            
            # Remove rows with null values in sort column
            data = data.dropna(subset=[sort_column])
            
            if data.empty:
                return _error_response(f"No data available for sorting by {sort_by}", "client_error")
            
            # Sort by specified metric
            ascending = (order.lower() == 'asc')
            sorted_data = data.sort_values(by=sort_column, ascending=ascending).head(limit)
            
            # Create leaderboard
            leaderboard = []
            for rank, (idx, row) in enumerate(sorted_data.iterrows(), 1):
                # Get game_pk for video links
                game_pk = row.get('game_pk')

                # Generate video-related URLs
                video_info = {}
                if game_pk:
                    video_info['game_highlights_url'] = f"https://www.mlb.com/gameday/{game_pk}/video"

                    pitcher_name = str(row.get('player_name', '')).replace(' ', '+')
                    game_date = str(row.get('game_date', ''))
                    if pitcher_name and game_date:
                        video_info['film_room_search'] = f"https://www.mlb.com/video/search?q={pitcher_name}+{game_date}"

                    video_info['game_pk'] = str(game_pk)
                    video_info['api_highlights_endpoint'] = f"https://statsapi.mlb.com/api/v1/schedule?gamePk={game_pk}&hydrate=game(content(highlights(highlights)))"

                entry = {
                    "rank": rank,
                    "player": str(row.get('batter_name', f"Player {row.get('batter', 'Unknown')}")),
                    "pitcher": str(row.get('player_name', 'Unknown')),  # player_name is the pitcher
                    "date": str(row.get('game_date', 'Unknown')),
                    "exit_velocity": float(row.get('launch_speed')) if pd.notna(row.get('launch_speed')) else None,
                    "launch_angle": float(row.get('launch_angle')) if pd.notna(row.get('launch_angle')) else None,
                    "distance": float(row.get('hit_distance_sc')) if pd.notna(row.get('hit_distance_sc')) else None,
                    "result": str(row.get('events', 'Unknown')),
                    "pitch_velocity": float(row.get('release_speed')) if pd.notna(row.get('release_speed')) else None,
                    "pitch_type": str(row.get('pitch_type', 'Unknown')),
                    "spin_rate": float(row.get('release_spin_rate')) if pd.notna(row.get('release_spin_rate')) else None,
                    "xba": float(row.get('estimated_ba_using_speedangle')) if pd.notna(row.get('estimated_ba_using_speedangle')) else None,
                    "xwoba": float(row.get('estimated_woba_using_speedangle')) if pd.notna(row.get('estimated_woba_using_speedangle')) else None,
                    "barrel": bool(row.get('barrel') == 1) if pd.notna(row.get('barrel')) else None,
                    "team": str(row.get('batting_team', 'Unknown')),
                    "description": str(row.get('des', 'No description')),
                    "video_links": video_info if video_info else None
                }
                leaderboard.append(entry)
            
            # Create response
            response = json.dumps({
                "start_date": start_date,
                "end_date": end_date,
                "filter": {
                    "result": result,
                    "min_ev": min_ev,
                    "min_pitch_velo": min_pitch_velo,
                    "pitch_type": pitch_type,
                    "player_name": player_name
                },
                "sorted_by": sort_by,
                "leaderboard": leaderboard
            }, indent=2, default=str)
            
            _cache_put(cache_key, response)
            return response

        except Exception as e:
            logger.error(f"Error getting statcast leaderboard: {str(e)}")
            return _error_response(str(e), "external_error")
    
    @mcp.tool()
    async def player_statcast(player_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                             pitch_type: Optional[str] = None, result_type: Optional[str] = None,
                             min_exit_velocity: Optional[Any] = None, min_distance: Optional[Any] = None) -> str:
        """
        Get comprehensive Statcast data for a specific player with advanced filtering.
    
        Args:
            player_name: Player name (e.g., 'Aaron Judge', 'Shohei Ohtani')
            start_date: Start date in YYYY-MM-DD format (optional, defaults to current season)
            end_date: End date in YYYY-MM-DD format (optional, defaults to current season)
            pitch_type: Filter by pitch type - 'FF' (4-seam), 'SL' (slider), 'CH' (changeup), 'CU' (curveball), 'SI' (sinker), 'FC' (cutter), 'FS' (splitter) (optional)
            result_type: Filter by result - 'home_run', 'hit', 'single', 'double', 'triple', 'batted_ball' (optional)
            min_exit_velocity: Minimum exit velocity in mph (optional)
            min_distance: Minimum distance in feet (optional)
    
        Returns:
            JSON with player stats, counts by pitch type, and top examples
        """
        try:
            player_name = sanitize_player_name(player_name)
            if start_date:
                start_date = validate_date(start_date, "start_date")
            if end_date:
                end_date = validate_date(end_date, "end_date")
        except ValueError as e:
            return _error_response(str(e), "client_error")

        try:
            if min_exit_velocity is not None:
                min_exit_velocity = float(min_exit_velocity)
            if min_distance is not None:
                min_distance = float(min_distance)
        except (ValueError, TypeError) as e:
            return _error_response(f"Invalid numeric filter: {e}", "client_error")

        try:
            player_id = await asyncio.to_thread(lookup_player_id, player_name)
        except LookupError as e:
            return _error_response(str(e), "client_error")

        try:
            from pybaseball import statcast_batter
            import pandas as pd

            if not start_date or not end_date:
                current_year = datetime.now().year
                start_date = start_date or f"{current_year}-04-01"
                end_date = end_date or datetime.now().strftime('%Y-%m-%d')
    
            logger.info(f"Fetching statcast data for {player_name} from {start_date} to {end_date}")
    
            data = await _fetch_with_retry(statcast_batter, start_date, end_date, player_id)
    
            if data.empty:
                return json.dumps({
                    "player": player_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "message": "No statcast data found for this player in the specified date range"
                })
    
            # Apply filters
            filtered_data = data.copy()
    
            if pitch_type:
                filtered_data = filtered_data[filtered_data['pitch_type'] == pitch_type]
    
            if result_type:
                if result_type == 'batted_ball':
                    filtered_data = filtered_data[filtered_data['type'] == 'X']
                elif result_type == 'hit':
                    filtered_data = filtered_data[filtered_data['events'].isin(['single', 'double', 'triple', 'home_run'])]
                else:
                    filtered_data = filtered_data[filtered_data['events'] == result_type]
    
            if min_exit_velocity:
                filtered_data = filtered_data[filtered_data['launch_speed'] >= min_exit_velocity]
    
            if min_distance:
                filtered_data = filtered_data[filtered_data['hit_distance_sc'] >= min_distance]
    
            # Calculate stats
            total_events = len(filtered_data)
    
            stats = {
                "player": player_name,
                "start_date": start_date,
                "end_date": end_date,
                "filters": {
                    "pitch_type": pitch_type,
                    "result_type": result_type,
                    "min_exit_velocity": min_exit_velocity,
                    "min_distance": min_distance
                },
                "total_events": total_events
            }
    
            if total_events == 0:
                stats["message"] = "No events matching the specified criteria"
                return json.dumps(stats, indent=2)
    
            # Overall statistics
            stats["overall_stats"] = {
                "avg_exit_velocity": float(filtered_data['launch_speed'].mean()) if 'launch_speed' in filtered_data.columns else None,
                "max_exit_velocity": float(filtered_data['launch_speed'].max()) if 'launch_speed' in filtered_data.columns else None,
                "avg_distance": float(filtered_data['hit_distance_sc'].mean()) if 'hit_distance_sc' in filtered_data.columns else None,
                "max_distance": float(filtered_data['hit_distance_sc'].max()) if 'hit_distance_sc' in filtered_data.columns else None,
                "avg_launch_angle": float(filtered_data['launch_angle'].mean()) if 'launch_angle' in filtered_data.columns else None,
                "barrel_count": int((filtered_data['barrel'] == 1).sum()) if 'barrel' in filtered_data.columns else 0,
                "barrel_rate": float((filtered_data['barrel'] == 1).sum() / len(filtered_data) * 100) if 'barrel' in filtered_data.columns and len(filtered_data) > 0 else 0
            }
    
            # Breakdown by pitch type (if not already filtered by pitch type)
            if not pitch_type and 'pitch_type' in filtered_data.columns:
                pitch_type_counts = filtered_data['pitch_type'].value_counts().to_dict()
                stats["by_pitch_type"] = {str(k): int(v) for k, v in pitch_type_counts.items()}
    
            # Breakdown by result type (if not already filtered)
            if not result_type and 'events' in filtered_data.columns:
                result_counts = filtered_data['events'].value_counts().head(10).to_dict()
                stats["by_result"] = {str(k): int(v) for k, v in result_counts.items()}
    
            # Top examples (by exit velocity)
            top_examples = []
            if 'launch_speed' in filtered_data.columns:
                top_data = filtered_data.nlargest(5, 'launch_speed')
    
                for _, row in top_data.iterrows():
                    example = {
                        'date': str(row.get('game_date', 'Unknown')),
                        'result': str(row.get('events', 'Unknown')),
                        'pitch_type': str(row.get('pitch_type', 'Unknown')),
                        'pitcher': str(row.get('player_name', 'Unknown')),
                        'exit_velocity': float(row.get('launch_speed')) if pd.notna(row.get('launch_speed')) else None,
                        'launch_angle': float(row.get('launch_angle')) if pd.notna(row.get('launch_angle')) else None,
                        'distance': float(row.get('hit_distance_sc')) if pd.notna(row.get('hit_distance_sc')) else None,
                        'pitch_velocity': float(row.get('release_speed')) if pd.notna(row.get('release_speed')) else None
                    }
    
                    # Add video links
                    game_pk = row.get('game_pk')
                    if game_pk:
                        example['video_url'] = f"https://www.mlb.com/gameday/{game_pk}/video"
    
                    top_examples.append(example)
    
            stats["top_examples"] = top_examples
    
            return json.dumps(stats, indent=2, default=str)
    
        except Exception as e:
            logger.error(f"Error in player_statcast: {str(e)}")
            return _error_response(str(e), "external_error")
    
    return mcp

# Module-level server instance for FastMCP CLI discovery
# FastMCP CLI looks for 'mcp', 'server', or 'app' at module level
mcp = create_server()

def main():
    """Main entry point for the MCP server.
    
    Transport mode is determined by environment variables:
    - If PORT is set (e.g., Railway), defaults to HTTP transport on 0.0.0.0:$PORT
    - Otherwise defaults to stdio transport for local/CLI usage
    
    Environment variables:
    - MCP_TRANSPORT: Override transport ('stdio', 'http', or 'sse')
    - MCP_HOST: Override host (default: '0.0.0.0' for HTTP)
    - MCP_PORT: Override port (default: $PORT or 8000)
    - MCP_PATH: Override MCP endpoint path (default: '/mcp')
    """
    logger.info("Starting MLB Stats MCP server")
    
    port_env = os.environ.get("PORT")
    transport = os.environ.get("MCP_TRANSPORT")
    
    if transport:
        transport = transport.lower()
    elif port_env:
        transport = "http"
    else:
        transport = "stdio"
    
    if transport in ("http", "sse"):
        host = os.environ.get("MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("MCP_PORT") or port_env or "8000")
        path = os.environ.get("MCP_PATH", "/mcp")
        logger.info(f"Running MCP server with {transport} transport on {host}:{port}{path}")
        mcp.run(transport=transport, host=host, port=port, path=path)
    else:
        logger.info("Running MCP server with stdio transport")
        mcp.run()

if __name__ == "__main__":
    main()
