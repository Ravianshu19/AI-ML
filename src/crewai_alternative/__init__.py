"""CrewAI alternative implementation of the research workflow.

This module provides an equivalent multi-step research pipeline
using CrewAI instead of LangGraph, demonstrating how the same
workflow can be expressed with a different agentic framework.

Usage:
    pip install crewai crewai-tools
    python -m src.crewai_alternative.crew
"""
from src.crewai_alternative.crew import run_crewai_research

__all__ = ["run_crewai_research"]
