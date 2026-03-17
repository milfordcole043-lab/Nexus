"""Entry point for: python -m nexus.mcp"""

from nexus.mcp.server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
