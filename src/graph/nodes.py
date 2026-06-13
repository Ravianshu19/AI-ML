"""Node functions for the LangGraph research workflow.

Each node is a function that takes the current ResearchState,
performs some operation (often involving LLM calls or tool use),
and returns a partial state update.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import get_llm, config
from src.tools.search import tavily_search_tool, tavily_search_results
from src.tools.scraper import scrape_webpage_tool
from src.tools.analyzer import (
    extract_key_facts_tool,
    compare_sources_tool,
    generate_citations_tool,
)
from src.graph.state import ResearchState

logger = logging.getLogger(__name__)

# ─── All tools available to the research agent ───────────────────────
RESEARCH_TOOLS = [
    tavily_search_tool,
    tavily_search_results,
    scrape_webpage_tool,
    extract_key_facts_tool,
    compare_sources_tool,
    generate_citations_tool,
]


# ─── 1. PLANNING NODE ───────────────────────────────────────────────
def plan_research(state: ResearchState) -> dict[str, Any]:
    """Generate a research plan with targeted sub-queries.

    Takes the user's broad research query and decomposes it into
    specific, searchable sub-queries for comprehensive coverage.
    """
    logger.info("📋 Planning research for: %s", state.query)

    llm = get_llm()
    planning_prompt = f"""You are a research planning assistant. Given the following research topic,
generate 3-5 specific search queries that will help gather comprehensive information.

Research Topic: {state.query}

Requirements:
- Each query should target a different aspect of the topic
- Include queries for: background/overview, recent developments, expert opinions, data/statistics
- Make queries specific and searchable

Return ONLY a JSON array of query strings, like:
["query 1", "query 2", "query 3"]"""

    response = llm.invoke([HumanMessage(content=planning_prompt)])

    # Parse the queries from the response
    try:
        # Try to extract JSON array from the response
        content = response.content
        # Find JSON array in the response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            queries = json.loads(content[start:end])
        else:
            queries = [state.query]  # Fallback to original query
    except (json.JSONDecodeError, ValueError):
        queries = [state.query]

    logger.info("📋 Generated %d sub-queries", len(queries))

    return {
        "search_queries": queries,
        "current_step": "researching",
        "messages": [
            AIMessage(content=f"Research plan created with {len(queries)} sub-queries: {queries}")
        ],
    }


# ─── 2. RESEARCH NODE (with tool calling) ────────────────────────────
def execute_research(state: ResearchState) -> dict[str, Any]:
    """Execute research by searching and scraping for each sub-query.

    Uses the LLM with bound tools to perform web searches and
    optionally scrape pages for more detail.
    """
    logger.info(
        "🔍 Executing research — iteration %d/%d",
        state.iteration + 1,
        state.max_iterations,
    )

    llm = get_llm()
    llm_with_tools = llm.bind_tools(RESEARCH_TOOLS)

    # Build the research prompt
    queries_text = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(state.search_queries))
    existing_sources = len(state.raw_sources)

    research_prompt = f"""You are a thorough research agent. Search for information using the available tools.

Research Topic: {state.query}

Sub-queries to investigate:
{queries_text}

Sources already collected: {existing_sources}
Current iteration: {state.iteration + 1} / {state.max_iterations}

Instructions:
1. Use tavily_search_results to search for each sub-query
2. If a search result looks especially relevant, use scrape_webpage_tool to get more detail
3. Focus on finding factual, up-to-date information
4. Aim to gather diverse perspectives

Perform your searches now."""

    messages = [
        SystemMessage(content="You are a research agent with access to web search and scraping tools."),
        HumanMessage(content=research_prompt),
    ]

    # Run the agent loop — invoke LLM, execute tools, repeat
    collected_sources: list[dict[str, Any]] = []
    tool_map = {t.name: t for t in RESEARCH_TOOLS}
    errors: list[str] = []

    for step in range(config.workflow.max_tool_calls_per_step):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break  # LLM decided it has enough information

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            logger.info("  🔧 Calling tool: %s", tool_name)

            try:
                tool_fn = tool_map.get(tool_name)
                if tool_fn is None:
                    raise ValueError(f"Unknown tool: {tool_name}")

                result = tool_fn.invoke(tool_args)

                # Store structured results from search
                if tool_name == "tavily_search_results" and isinstance(result, list):
                    for item in result:
                        collected_sources.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "content": item.get("content", ""),
                            "source_tool": tool_name,
                        })
                elif tool_name == "scrape_webpage_tool":
                    collected_sources.append({
                        "title": f"Scraped: {tool_args.get('url', 'unknown')}",
                        "url": tool_args.get("url", ""),
                        "content": str(result),
                        "source_tool": tool_name,
                    })

                # Feed result back to LLM
                from langchain_core.messages import ToolMessage
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    )
                )

            except Exception as e:
                error_msg = f"Error calling {tool_name}: {e}"
                logger.error("  ❌ %s", error_msg)
                errors.append(error_msg)
                from langchain_core.messages import ToolMessage
                messages.append(
                    ToolMessage(
                        content=f"Error: {e}",
                        tool_call_id=tool_call["id"],
                    )
                )

    logger.info("🔍 Collected %d new sources", len(collected_sources))

    return {
        "raw_sources": collected_sources,
        "iteration": state.iteration + 1,
        "current_step": "analyzing",
        "errors": errors,
        "messages": [
            AIMessage(
                content=f"Research iteration {state.iteration + 1} complete. "
                f"Collected {len(collected_sources)} new sources."
            )
        ],
    }


# ─── 3. ANALYSIS NODE ────────────────────────────────────────────────
def analyze_findings(state: ResearchState) -> dict[str, Any]:
    """Analyze and cross-reference collected research findings.

    Synthesizes information from all sources, identifies key themes,
    and assesses whether more research is needed.
    """
    logger.info("🧪 Analyzing %d total sources", len(state.raw_sources))

    llm = get_llm()

    # Compile source summaries
    source_summaries = []
    for i, source in enumerate(state.raw_sources, 1):
        content_preview = source.get("content", "")[:500]
        source_summaries.append(
            f"Source {i} [{source.get('title', 'Untitled')}]:\n{content_preview}"
        )

    sources_text = "\n\n---\n\n".join(source_summaries)

    analysis_prompt = f"""Analyze the following research sources about: {state.query}

Collected Sources:
{sources_text}

Perform the following analysis:
1. **Key Findings**: List the most important facts and insights (bullet points)
2. **Consensus**: What do multiple sources agree on?
3. **Gaps**: What aspects of the topic are NOT well-covered by these sources?
4. **Confidence Assessment**: How confident are you in the findings? (High/Medium/Low)
5. **Need More Research?**: Based on the gaps, should we do another round of research? Answer YES or NO.

Provide your analysis in a structured format."""

    response = llm.invoke([HumanMessage(content=analysis_prompt)])
    analysis_text = response.content

    # Determine if we should continue researching
    should_continue = (
        "YES" in analysis_text.upper().split("NEED MORE RESEARCH")[-1][:50]
        if "NEED MORE RESEARCH" in analysis_text.upper()
        else False
    )

    # Don't continue if we've hit max iterations
    if state.iteration >= state.max_iterations:
        should_continue = False

    logger.info(
        "🧪 Analysis complete — continue researching: %s", should_continue
    )

    return {
        "analysis": analysis_text,
        "should_continue": should_continue,
        "current_step": "deciding" if should_continue else "writing",
        "messages": [
            AIMessage(
                content=f"Analysis complete. Continue: {should_continue}"
            )
        ],
    }


# ─── 4. REPORT WRITING NODE ─────────────────────────────────────────
def write_report(state: ResearchState) -> dict[str, Any]:
    """Generate the final research report from analyzed findings.

    Produces a well-structured, comprehensive report with proper
    citations and clear conclusions.
    """
    logger.info("📝 Writing final report")

    llm = get_llm()

    # Compile all available information
    source_refs = []
    for i, source in enumerate(state.raw_sources, 1):
        source_refs.append(
            f"[{i}] {source.get('title', 'Untitled')} — {source.get('url', 'N/A')}"
        )

    report_prompt = f"""Write a comprehensive research report based on the following:

Original Query: {state.query}

Analysis:
{state.analysis}

Available Sources:
{chr(10).join(source_refs)}

Requirements:
1. Start with a clear executive summary
2. Organize findings into logical sections with headers
3. Include specific data, statistics, and quotes where available
4. Note any limitations or areas of uncertainty
5. End with clear conclusions and potential next steps
6. Reference sources using [N] notation

Write a professional, thorough report in Markdown format."""

    response = llm.invoke([HumanMessage(content=report_prompt)])

    # Generate citations
    citations_input = [
        {
            "title": s.get("title", "Untitled"),
            "url": s.get("url", "N/A"),
        }
        for s in state.raw_sources
    ]
    citations = generate_citations_tool.invoke(
        {"sources": citations_input}
    )

    logger.info("📝 Report complete — %d words", len(response.content.split()))

    return {
        "report": response.content,
        "citations": citations,
        "current_step": "complete",
        "messages": [
            AIMessage(content="Final research report generated.")
        ],
    }


# ─── 5. ROUTING FUNCTION ────────────────────────────────────────────
def should_continue_research(state: ResearchState) -> str:
    """Conditional edge: decide whether to loop back for more research.

    Returns:
        'research' to do another research loop, or 'report' to write the final report.
    """
    if state.should_continue and state.iteration < state.max_iterations:
        logger.info("🔄 Routing: back to research (iteration %d)", state.iteration + 1)
        return "research"
    else:
        logger.info("✅ Routing: proceed to report writing")
        return "report"
