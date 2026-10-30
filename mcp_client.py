"""
mcp_client.py
-------------
Optional real MCP (Model Context Protocol) integration. If the operator
sets MCP_SERVER_URL (any streamable-HTTP MCP server that exposes a
search-like tool — e.g. a Tavily/Exa/Brave MCP endpoint), the Research
Agent will call it over the actual MCP protocol instead of the plain
REST fallbacks in research_tools.py.

This is intentionally separate from research_tools.py: if the `mcp`
package isn't installed, or MCP_SERVER_URL isn't set, or the server is
unreachable, everything here fails soft and the orchestrator silently
keeps using the REST fallbacks. Nothing else in the app depends on MCP
being available.
"""

import os
from typing import Optional

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")  # e.g. https://mcp.tavily.com/mcp/?tavilyApiKey=...
MCP_TOOL_HINT = os.getenv("MCP_SEARCH_TOOL_NAME")  # optional: force a specific tool name

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    MCP_AVAILABLE = True
except Exception:
    MCP_AVAILABLE = False


def mcp_configured() -> bool:
    return MCP_AVAILABLE and bool(MCP_SERVER_URL)


async def mcp_search(query: str, max_results: int = 5) -> Optional[list[dict]]:
    """Returns [{title, url, snippet}, ...] via a live MCP tool call, or
    None if MCP isn't configured / the call failed (caller should fall
    back to research_tools.web_search in that case)."""
    if not mcp_configured():
        return None

    try:
        async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_resp = await session.list_tools()
                tool_name = MCP_TOOL_HINT
                if not tool_name:
                    for t in tools_resp.tools:
                        name = (t.name or "").lower()
                        if "search" in name or "web" in name:
                            tool_name = t.name
                            break
                if not tool_name:
                    return None

                result = await session.call_tool(
                    tool_name, arguments={"query": query, "max_results": max_results}
                )
                if result.isError:
                    return None

                items: list[dict] = []
                for block in result.content:
                    text = getattr(block, "text", None)
                    if text:
                        items.append({"title": "", "url": "", "snippet": text[:500]})
                return items[:max_results] if items else None
    except Exception:
        return None