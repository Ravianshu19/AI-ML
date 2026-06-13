"""Web search tools using Tavily API."""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.tools import tool
from tavily import TavilyClient

from src.config import config


def _get_tavily_client() -> TavilyClient:
    """Lazily create a Tavily client."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required.")
    return TavilyClient(api_key=api_key)


@tool
def tavily_search_tool(query: str) -> str:
    """Search the web for information on a given query.

    Use this tool to find current, up-to-date information about any topic.
    Returns a concise answer synthesized from multiple web sources.

    Args:
        query: The search query string.

    Returns:
        A synthesized answer from web search results.
    """
    client = _get_tavily_client()
    response = client.search(
        query=query,
        search_depth=config.search.search_depth,
        max_results=config.search.max_results,
        include_answer=True,
    )
    return response.get("answer", "No answer found.")


@tool
def tavily_search_results(
    query: str,
    max_results: Optional[int] = None,
) -> list[dict]:
    """Search the web and return detailed results with URLs and content.

    Use this tool when you need specific sources, URLs, and raw content
    from web search results for deeper analysis.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 5).

    Returns:
        A list of search result dictionaries containing title, url, and content.
    """
    client = _get_tavily_client()
    response = client.search(
        query=query,
        search_depth=config.search.search_depth,
        max_results=max_results or config.search.max_results,
        include_raw_content=False,
    )
    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        })
    return results
