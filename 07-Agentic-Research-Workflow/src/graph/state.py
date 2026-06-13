"""State definitions for the LangGraph research workflow.

The state is the central data structure that flows through every node
in the graph. It accumulates research findings, tracks progress, and
holds the final output.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


@dataclass
class ResearchState:
    """Shared state for the multi-step research workflow.

    Attributes:
        messages: Conversation / reasoning messages (appended via reducer).
        query: The original user research query.
        search_queries: Generated sub-queries for multi-angle research.
        raw_sources: Collected raw search results and scraped content.
        key_facts: Extracted key facts from sources.
        analysis: Intermediate analysis and cross-referencing results.
        report: The final synthesized research report.
        citations: Formatted citations for the report.
        current_step: Tracks which step the workflow is on.
        iteration: Current research loop iteration.
        max_iterations: Maximum allowed research loops.
        should_continue: Whether the research loop should continue.
        errors: Any errors encountered during execution.
    """

    # Messages use the `add_messages` reducer to append, not overwrite
    messages: Annotated[list[BaseMessage], add_messages] = field(
        default_factory=list
    )

    # Input
    query: str = ""

    # Research planning
    search_queries: list[str] = field(default_factory=list)

    # Data collection
    raw_sources: Annotated[list[dict[str, Any]], operator.add] = field(
        default_factory=list
    )

    # Analysis
    key_facts: list[str] = field(default_factory=list)
    analysis: str = ""

    # Output
    report: str = ""
    citations: str = ""

    # Control flow
    current_step: str = "planning"
    iteration: int = 0
    max_iterations: int = 3
    should_continue: bool = True

    # Error tracking
    errors: Annotated[list[str], operator.add] = field(default_factory=list)
