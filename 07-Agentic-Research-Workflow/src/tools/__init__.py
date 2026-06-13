"""Custom tools for the research agent."""
from src.tools.search import tavily_search_tool, tavily_search_results
from src.tools.scraper import scrape_webpage_tool
from src.tools.analyzer import (
    extract_key_facts_tool,
    compare_sources_tool,
    generate_citations_tool,
)

__all__ = [
    "tavily_search_tool",
    "tavily_search_results",
    "scrape_webpage_tool",
    "extract_key_facts_tool",
    "compare_sources_tool",
    "generate_citations_tool",
]
