"""Analysis tools for processing and synthesizing research data."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def extract_key_facts_tool(text: str, topic: str) -> str:
    """Extract key facts and claims from a body of text related to a topic.

    Use this tool to distill important facts, statistics, and claims
    from a large body of text, focusing on what is relevant to the topic.

    Args:
        text: The text to extract facts from.
        topic: The topic to focus fact extraction on.

    Returns:
        A structured list of key facts extracted from the text.
    """
    # This is a "structured extraction" tool — the LLM calling this
    # will incorporate the text + topic in its reasoning. The tool
    # itself serves as a structured interface for the agent to use.
    lines = text.strip().split("\n")
    facts = []
    keywords = topic.lower().split()

    for line in lines:
        line_lower = line.lower()
        # Heuristic: keep lines that mention topic keywords or contain numbers
        if any(kw in line_lower for kw in keywords) or any(c.isdigit() for c in line):
            cleaned = line.strip()
            if len(cleaned) > 20:  # Skip very short lines
                facts.append(f"• {cleaned}")

    if not facts:
        return f"No specific facts found for topic '{topic}' in the provided text."

    return f"Key facts related to '{topic}':\n" + "\n".join(facts[:20])


@tool
def compare_sources_tool(source_texts: list[str], topic: str) -> str:
    """Compare information across multiple sources to identify consensus and conflicts.

    Use this tool when you have gathered information from multiple sources
    and need to cross-reference them for accuracy and completeness.

    Args:
        source_texts: A list of text excerpts from different sources.
        topic: The topic being researched.

    Returns:
        A comparison analysis of the sources.
    """
    if not source_texts:
        return "No sources provided for comparison."

    analysis_parts = [f"## Source Comparison for: {topic}\n"]
    analysis_parts.append(f"**Number of sources analyzed:** {len(source_texts)}\n")

    for i, text in enumerate(source_texts, 1):
        word_count = len(text.split())
        preview = text[:200].replace("\n", " ")
        analysis_parts.append(
            f"### Source {i}\n"
            f"- **Word count:** {word_count}\n"
            f"- **Preview:** {preview}...\n"
        )

    analysis_parts.append(
        "\n**Note:** The agent should synthesize these sources, "
        "identify agreements and contradictions, and produce a "
        "unified analysis."
    )

    return "\n".join(analysis_parts)


@tool
def generate_citations_tool(
    sources: list[dict],
) -> str:
    """Generate properly formatted citations from a list of sources.

    Use this tool to create a formatted bibliography/reference list
    from the sources gathered during research.

    Args:
        sources: List of source dictionaries with 'title', 'url', and optionally 'date' keys.

    Returns:
        A formatted citation list.
    """
    if not sources:
        return "No sources provided for citation generation."

    citations = ["## References\n"]
    for i, source in enumerate(sources, 1):
        title = source.get("title", "Untitled")
        url = source.get("url", "No URL")
        date = source.get("date", "n.d.")
        citations.append(f"[{i}] {title}. Retrieved {date}. {url}")

    return "\n".join(citations)
