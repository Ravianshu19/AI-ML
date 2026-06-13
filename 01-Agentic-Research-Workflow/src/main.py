"""Main entry point for the Agentic Research Workflow.

Provides a CLI interface using Typer for running research queries
with full observability and beautiful terminal output via Rich.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.theme import Theme

# ─── App Setup ────────────────────────────────────────────────────────
custom_theme = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "heading": "bold magenta",
    }
)

console = Console(theme=custom_theme)
app = typer.Typer(
    name="research",
    help="🔬 Agentic Research Workflow — Multi-step research with LangGraph",
    add_completion=False,
)


def _setup_logging(verbose: bool = False) -> None:
    """Configure structured logging with Rich."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False,
            )
        ],
    )


# ─── Main Research Command ───────────────────────────────────────────
@app.command()
def research(
    query: str = typer.Argument(
        ...,
        help="The research question or topic to investigate.",
    ),
    max_iterations: int = typer.Option(
        3,
        "--max-iterations",
        "-n",
        help="Maximum number of research loops.",
        min=1,
        max=10,
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the report as a Markdown file.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose/debug logging.",
    ),
    provider: str = typer.Option(
        None,
        "--observability",
        "--obs",
        help="Observability provider: langsmith, langfuse, both, none.",
    ),
) -> None:
    """Run a multi-step research workflow on the given topic.

    The agent will:
    1. 📋 Plan — decompose the query into targeted sub-queries
    2. 🔍 Research — search the web and scrape relevant pages
    3. 🧪 Analyze — cross-reference sources and assess coverage
    4. 🔄 Loop — repeat research if gaps are found
    5. 📝 Report — generate a comprehensive Markdown report
    """
    _setup_logging(verbose)

    # Override observability provider if specified
    if provider:
        import os
        os.environ["OBSERVABILITY_PROVIDER"] = provider

    # Load config (after env override)
    from src.config import config as app_config

    console.print()
    console.print(
        Panel(
            f"[heading]🔬 Agentic Research Workflow[/heading]\n\n"
            f'[info]Query:[/info]  "{query}"\n'
            f"[info]Max iterations:[/info]  {max_iterations}\n"
            f"[info]Observability:[/info]  {app_config.observability.value}\n"
            f"[info]LLM:[/info]  {app_config.llm.provider} / {app_config.llm.model}",
            title="Configuration",
            border_style="cyan",
        )
    )
    console.print()

    # Run the workflow
    try:
        from src.graph.workflow import run_research as _run_research

        result = asyncio.run(
            _run_research(
                query=query,
                max_iterations=max_iterations,
            )
        )

        # Display results
        _display_results(result, output)

    except KeyboardInterrupt:
        console.print("\n[warning]⚠️  Research interrupted by user.[/warning]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[error]❌ Error: {e}[/error]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


def _display_results(result: dict, output_path: str | None = None) -> None:
    """Display the research results with Rich formatting."""
    console.print()
    console.rule("[success]✅ Research Complete[/success]")
    console.print()

    # Show report
    report = result.get("report", "No report generated.")
    console.print(
        Panel(
            Markdown(report),
            title="📄 Research Report",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Show citations
    citations = result.get("citations", "")
    if citations:
        console.print()
        console.print(
            Panel(
                Markdown(citations),
                title="📚 Citations",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Show stats
    sources_count = len(result.get("raw_sources", []))
    iterations = result.get("iteration", 0)
    errors = result.get("errors", [])

    console.print()
    console.print(
        Panel(
            f"[info]Sources collected:[/info]  {sources_count}\n"
            f"[info]Research iterations:[/info]  {iterations}\n"
            f"[info]Errors encountered:[/info]  {len(errors)}",
            title="📊 Statistics",
            border_style="cyan",
        )
    )

    if errors:
        console.print()
        for err in errors:
            console.print(f"  [warning]⚠️  {err}[/warning]")

    # Save to file if requested
    if output_path:
        path = Path(output_path)
        full_report = f"{report}\n\n---\n\n{citations}"
        path.write_text(full_report, encoding="utf-8")
        console.print(f"\n[success]💾 Report saved to: {path.absolute()}[/success]")


# ─── Graph Visualization Command ────────────────────────────────────
@app.command()
def visualize() -> None:
    """Display the research workflow graph structure."""
    console.print()
    console.print(
        Panel(
            """
[heading]Research Workflow Graph[/heading]

    START
      │
      ▼
  ┌──────────┐
  │   PLAN   │  ← Decompose query into sub-queries
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ RESEARCH │  ← Search & scrape (with tool calling)
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ ANALYZE  │  ← Cross-reference & assess coverage
  └────┬─────┘
       │
       ├─── should_continue? ──► YES ──► back to RESEARCH
       │
       ▼ NO
  ┌──────────┐
  │  REPORT  │  ← Write final Markdown report
  └────┬─────┘
       │
       ▼
      END
""",
            title="🗺️  Workflow Graph",
            border_style="magenta",
        )
    )


# ─── Config Info Command ────────────────────────────────────────────
@app.command()
def config_info() -> None:
    """Display the current configuration."""
    from src.config import config as app_config

    console.print()
    console.print(
        Panel(
            f"[heading]LLM[/heading]\n"
            f"  Provider:     {app_config.llm.provider}\n"
            f"  Model:        {app_config.llm.model}\n"
            f"  Temperature:  {app_config.llm.temperature}\n"
            f"  Max tokens:   {app_config.llm.max_tokens}\n\n"
            f"[heading]Search[/heading]\n"
            f"  Provider:     {app_config.search.provider}\n"
            f"  Max results:  {app_config.search.max_results}\n"
            f"  Depth:        {app_config.search.search_depth}\n\n"
            f"[heading]Workflow[/heading]\n"
            f"  Max loops:    {app_config.workflow.max_research_loops}\n"
            f"  Max tools/step: {app_config.workflow.max_tool_calls_per_step}\n"
            f"  Parallel:     {app_config.workflow.parallel_tool_calls}\n\n"
            f"[heading]Observability[/heading]\n"
            f"  Provider:     {app_config.observability.value}",
            title="⚙️  Configuration",
            border_style="cyan",
        )
    )


# ─── Entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    app()
