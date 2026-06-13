"""Web scraping tool for extracting content from URLs."""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool
def scrape_webpage_tool(url: str) -> str:
    """Scrape and extract the main text content from a webpage URL.

    Use this tool to get the full text content of a specific webpage
    when you need more detail than what search results provide.

    Args:
        url: The full URL of the webpage to scrape.

    Returns:
        The extracted text content from the webpage (truncated to 8000 chars).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
    except httpx.HTTPError as e:
        return f"Error fetching {url}: {e}"

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # Extract text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    content = "\n".join(lines)

    # Truncate to avoid token limits
    max_chars = 8000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[... content truncated ...]"

    return content
