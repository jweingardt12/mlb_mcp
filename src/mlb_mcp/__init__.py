"""MLB Stats MCP Server package."""

__all__ = ["create_server", "main"]


def __getattr__(name: str):
    """Lazy import to avoid RuntimeWarning when running as module."""
    if name in __all__:
        from mlb_mcp import server
        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
