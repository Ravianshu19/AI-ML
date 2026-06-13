"""LangGraph workflow assembly and execution.

This module wires together all the nodes and edges into a
compilable, executable LangGraph StateGraph.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import ResearchState
from src.graph.nodes import (
    plan_research,
    execute_research,
    analyze_findings,
    write_report,
    should_continue_research,
)
from src.observability import initialize_observability

logger = logging.getLogger(__name__)


def build_research_graph() -> StateGraph:
    """Build and compile the multi-step research workflow graph.

    Graph structure:

        START
          │
          ▼
      ┌──────────┐
      │  PLAN    │  ← Decompose query into sub-queries
      └────┬─────┘
           │
           ▼
      ┌──────────┐
      │ RESEARCH │  ← Search & scrape (with tool calling)
      └────┬─────┘
           │
           ▼
      ┌──────────┐
      │ ANALYZE  │  ← Cross-reference & assess
      └────┬─────┘
           │
           ├─── should_continue? ──► YES ──► back to RESEARCH
           │
           ▼ NO
      ┌──────────┐
      │  REPORT  │  ← Write final report
      └────┬─────┘
           │
           ▼
          END

    Returns:
        A compiled LangGraph StateGraph ready for execution.
    """
    # Build the graph
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("plan", plan_research)
    graph.add_node("research", execute_research)
    graph.add_node("analyze", analyze_findings)
    graph.add_node("report", write_report)

    # Add edges
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "analyze")

    # Conditional edge: loop or proceed
    graph.add_conditional_edges(
        "analyze",
        should_continue_research,
        {
            "research": "research",  # Loop back
            "report": "report",      # Proceed to report
        },
    )

    graph.add_edge("report", END)

    # Compile with checkpointing for state persistence
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("✅ Research graph compiled successfully")
    return compiled


async def run_research(
    query: str,
    max_iterations: int = 3,
    thread_id: str = "default",
) -> dict[str, Any]:
    """Run the full research workflow for a given query.

    Args:
        query: The research question or topic.
        max_iterations: Maximum research loops (default: 3).
        thread_id: Thread ID for checkpointing.

    Returns:
        The final ResearchState as a dictionary.
    """
    # Initialize observability callbacks
    callbacks = initialize_observability()

    # Build the graph
    graph = build_research_graph()

    # Prepare initial state
    initial_state = ResearchState(
        query=query,
        max_iterations=max_iterations,
    )

    # Run with config
    run_config = {
        "configurable": {"thread_id": thread_id},
    }
    if callbacks:
        run_config["callbacks"] = callbacks

    logger.info("🚀 Starting research workflow for: %s", query)

    # Execute the graph
    final_state = await graph.ainvoke(
        initial_state,
        config=run_config,
    )

    logger.info("🏁 Research workflow complete")
    return final_state
