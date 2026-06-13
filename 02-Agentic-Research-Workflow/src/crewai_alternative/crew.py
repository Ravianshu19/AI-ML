"""CrewAI-based research workflow — alternative to the LangGraph implementation.

This module defines the same multi-step research pipeline using CrewAI's
Agent/Task/Crew abstractions. It demonstrates:
  - Agent role definitions with backstories
  - Task decomposition with expected outputs
  - Sequential crew execution with tool delegation
  - Langfuse/LangSmith observability integration

Prerequisites:
    pip install crewai crewai-tools langfuse langsmith
"""

from __future__ import annotations

import os
import logging
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ─── Observability Decorators ────────────────────────────────────────
def _get_langfuse_handler():
    """Get Langfuse callback handler if configured."""
    try:
        from langfuse.callback import CallbackHandler

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        if not public_key or not secret_key:
            return None

        return CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            tags=["crewai-research"],
        )
    except ImportError:
        return None


# ─── Custom Tools ────────────────────────────────────────────────────
def _build_tools():
    """Build the tool set for CrewAI agents.

    Uses crewai_tools for built-in capabilities plus our custom tools
    wrapped for CrewAI compatibility.
    """
    from crewai_tools import (
        SerperDevTool,
        ScrapeWebsiteTool,
    )
    from crewai import tool as crewai_tool

    tools = []

    # Web search tool
    try:
        search_tool = SerperDevTool()
        tools.append(search_tool)
    except Exception:
        logger.warning("SerperDevTool not available. Set SERPER_API_KEY.")

    # Web scraper tool
    scrape_tool = ScrapeWebsiteTool()
    tools.append(scrape_tool)

    # Custom analysis tool
    @crewai_tool("Extract Key Facts")
    def extract_facts(text: str, topic: str) -> str:
        """Extract key facts from text related to a specific topic.

        Args:
            text: The text to analyze.
            topic: The topic to focus on.

        Returns:
            Bullet-pointed list of key facts.
        """
        lines = text.strip().split("\n")
        keywords = topic.lower().split()
        facts = []
        for line in lines:
            lower = line.lower()
            if any(kw in lower for kw in keywords) or any(
                c.isdigit() for c in line
            ):
                cleaned = line.strip()
                if len(cleaned) > 20:
                    facts.append(f"• {cleaned}")
        return "\n".join(facts[:15]) if facts else "No specific facts found."

    tools.append(extract_facts)

    return tools


# ─── Agent Definitions ───────────────────────────────────────────────
def _create_agents(tools: list):
    """Create the specialized research agents.

    Returns:
        Tuple of (planner, researcher, analyst, writer) agents.
    """
    from crewai import Agent

    # 1. Research Planner Agent
    planner = Agent(
        role="Research Planner",
        goal=(
            "Decompose a broad research topic into specific, targeted "
            "sub-questions that ensure comprehensive coverage."
        ),
        backstory=(
            "You are a senior research strategist with 20 years of experience "
            "at top think tanks. You excel at breaking down complex topics into "
            "structured, actionable research plans. You understand that good "
            "research requires examining a topic from multiple angles: "
            "historical context, current state, expert opinions, data, and "
            "future trends."
        ),
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # 2. Research Agent (has tools)
    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            "Conduct thorough web research using search and scraping tools "
            "to gather high-quality, factual information from diverse sources."
        ),
        backstory=(
            "You are an expert research analyst skilled at finding needles in "
            "haystacks. You know how to craft effective search queries, "
            "evaluate source credibility, and extract the most relevant "
            "information. You always verify facts across multiple sources "
            "and note any discrepancies."
        ),
        verbose=True,
        tools=tools,
        allow_delegation=False,
        max_iter=10,
    )

    # 3. Analysis Agent
    analyst = Agent(
        role="Critical Analysis Specialist",
        goal=(
            "Synthesize research findings, identify patterns and gaps, "
            "cross-reference sources, and assess the overall quality and "
            "completeness of the gathered information."
        ),
        backstory=(
            "You are a data analyst with expertise in critical thinking and "
            "information synthesis. You excel at finding connections between "
            "disparate pieces of information, identifying biases in sources, "
            "and determining what's missing from a body of research. You "
            "always provide an honest assessment of confidence levels."
        ),
        verbose=True,
        tools=tools,
        allow_delegation=True,
        max_iter=5,
    )

    # 4. Report Writer Agent
    writer = Agent(
        role="Research Report Writer",
        goal=(
            "Transform analyzed research findings into a clear, professional, "
            "well-structured Markdown report with proper citations."
        ),
        backstory=(
            "You are an award-winning technical writer who transforms complex "
            "research into accessible, engaging reports. You know how to "
            "structure information for maximum clarity, use data effectively, "
            "and write compelling executive summaries. Your reports are known "
            "for being thorough yet readable."
        ),
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    return planner, researcher, analyst, writer


# ─── Task Definitions ────────────────────────────────────────────────
def _create_tasks(
    query: str,
    planner,
    researcher,
    analyst,
    writer,
):
    """Create the sequential tasks for the research workflow.

    Args:
        query: The research topic.
        planner: Planner agent.
        researcher: Researcher agent.
        analyst: Analyst agent.
        writer: Writer agent.

    Returns:
        List of Task objects in execution order.
    """
    from crewai import Task

    # Task 1: Planning
    planning_task = Task(
        description=(
            f"Create a comprehensive research plan for the following topic:\n\n"
            f'"{query}"\n\n'
            f"Decompose it into 3-5 specific sub-questions covering:\n"
            f"- Background and historical context\n"
            f"- Current state and recent developments\n"
            f"- Key players, experts, and organizations\n"
            f"- Data, statistics, and measurable outcomes\n"
            f"- Future outlook and emerging trends"
        ),
        expected_output=(
            "A structured research plan with:\n"
            "1. A clear statement of the research scope\n"
            "2. 3-5 specific, searchable sub-questions\n"
            "3. Priority order for investigation\n"
            "4. Expected types of sources for each question"
        ),
        agent=planner,
    )

    # Task 2: Research Execution
    research_task = Task(
        description=(
            "Using the research plan provided, conduct thorough web research.\n\n"
            "For each sub-question in the plan:\n"
            "1. Search the web using targeted queries\n"
            "2. Scrape the most promising results for detailed content\n"
            "3. Extract key facts, data points, and quotes\n"
            "4. Note the source URL and credibility indicators\n\n"
            "Gather information from at least 5 different sources. "
            "Prioritize recent, authoritative sources."
        ),
        expected_output=(
            "A comprehensive collection of research findings organized by "
            "sub-question, including:\n"
            "- Key facts and data points with source URLs\n"
            "- Direct quotes from experts or officials\n"
            "- Statistics and numerical data\n"
            "- Any conflicting information noted"
        ),
        agent=researcher,
        context=[planning_task],
    )

    # Task 3: Analysis
    analysis_task = Task(
        description=(
            "Analyze and synthesize all research findings.\n\n"
            "Perform the following:\n"
            "1. Cross-reference facts across multiple sources\n"
            "2. Identify consensus points and contradictions\n"
            "3. Assess source credibility and potential biases\n"
            "4. Identify gaps in the research coverage\n"
            "5. Rate confidence level (High/Medium/Low) for each finding\n"
            "6. Determine if additional research is needed"
        ),
        expected_output=(
            "A structured analysis report containing:\n"
            "- Verified key findings (with confidence levels)\n"
            "- Points of consensus across sources\n"
            "- Contradictions or disputes identified\n"
            "- Research gaps and limitations\n"
            "- Overall confidence assessment\n"
            "- Recommendation: whether more research is needed"
        ),
        agent=analyst,
        context=[planning_task, research_task],
    )

    # Task 4: Report Writing
    report_task = Task(
        description=(
            f'Write a comprehensive research report on: "{query}"\n\n'
            f"Use the research findings and analysis provided. Structure:\n\n"
            f"1. **Executive Summary** (2-3 paragraphs)\n"
            f"2. **Background** (context and history)\n"
            f"3. **Key Findings** (organized by theme)\n"
            f"4. **Analysis** (synthesis and implications)\n"
            f"5. **Limitations** (gaps and caveats)\n"
            f"6. **Conclusions & Next Steps**\n"
            f"7. **References** (numbered list with URLs)\n\n"
            f"Use Markdown formatting. Include [N] citation references."
        ),
        expected_output=(
            "A polished, professional Markdown research report of 1000-2000 "
            "words with proper structure, citations, and actionable conclusions."
        ),
        agent=writer,
        context=[planning_task, research_task, analysis_task],
    )

    return [planning_task, research_task, analysis_task, report_task]


# ─── Crew Assembly & Execution ───────────────────────────────────────
def run_crewai_research(
    query: str,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the CrewAI-based research workflow.

    Args:
        query: The research topic or question.
        verbose: Whether to enable verbose output.

    Returns:
        Dictionary with 'report' and 'metadata' keys.
    """
    from crewai import Crew, Process

    logger.info("🚀 Starting CrewAI research workflow")

    # Build components
    tools = _build_tools()
    planner, researcher, analyst, writer = _create_agents(tools)
    tasks = _create_tasks(query, planner, researcher, analyst, writer)

    # Assemble the crew
    crew = Crew(
        agents=[planner, researcher, analyst, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=verbose,
        # Memory for context sharing between agents
        memory=True,
        # Observability callbacks
        callbacks=_get_langfuse_handler(),
    )

    # Enable LangSmith tracing via env vars (auto-detected)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    # Execute the crew
    result = crew.kickoff()

    logger.info("🏁 CrewAI research workflow complete")

    return {
        "report": str(result),
        "metadata": {
            "agents": [a.role for a in crew.agents],
            "tasks_completed": len(tasks),
            "process": "sequential",
        },
    }


# ─── CLI Entry Point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    console = Console()

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What are the latest breakthroughs in AI agents and multi-agent systems?"
    )

    console.print()
    console.print(
        Panel(
            f"[bold magenta]🤖 CrewAI Research Workflow[/bold magenta]\n\n"
            f'[cyan]Query:[/cyan] "{query}"',
            border_style="magenta",
        )
    )
    console.print()

    result = run_crewai_research(query)

    console.print()
    console.rule("[green]✅ Research Complete[/green]")
    console.print()
    console.print(
        Panel(
            Markdown(result["report"]),
            title="📄 Research Report",
            border_style="green",
        )
    )
