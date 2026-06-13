"""End-to-end ML pipeline orchestrator.

Ties together all pipeline stages: feature generation, training,
evaluation, serving, and monitoring into a single runnable pipeline.

Usage:
    python -m pipeline --stage all
    python -m pipeline --stage train
    python -m pipeline --stage monitor --drift-level 0.3
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

console = Console()
app = typer.Typer(
    name="ml-pipeline",
    help="🏭 End-to-End ML Pipeline Orchestrator",
    add_completion=False,
)


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


@app.command()
def run(
    stage: str = typer.Option(
        "all",
        "--stage",
        "-s",
        help="Pipeline stage: features, train, evaluate, monitor, serve, all",
    ),
    model_type: str = typer.Option(
        "random_forest",
        "--model-type",
        "-m",
        help="Model type: random_forest, gradient_boosting, logistic_regression",
    ),
    drift_level: float = typer.Option(
        0.1,
        "--drift-level",
        "-d",
        help="Simulated drift level for monitoring (0.0-1.0)",
    ),
    register: bool = typer.Option(
        False,
        "--register",
        "-r",
        help="Register the trained model in MLflow",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the end-to-end ML pipeline.

    Stages:
      - features:  Generate synthetic data & populate feature store
      - train:     Train model with MLflow tracking
      - evaluate:  Compare runs & promote best model
      - monitor:   Run Evidently drift detection
      - serve:     Start the FastAPI model server
      - all:       Run features → train → evaluate → monitor
    """
    _setup_logging(verbose)

    console.print()
    console.print(
        Panel(
            "[bold cyan]🏭 End-to-End ML Platform[/bold cyan]\n\n"
            f"Stage:       {stage}\n"
            f"Model:       {model_type}\n"
            f"Drift level: {drift_level}\n"
            f"Register:    {register}",
            title="Pipeline Configuration",
            border_style="cyan",
        )
    )
    console.print()

    stages_to_run = (
        ["features", "train", "evaluate", "monitor"]
        if stage == "all"
        else [stage]
    )

    for current_stage in stages_to_run:
        console.rule(f"[bold magenta]Stage: {current_stage.upper()}[/bold magenta]")
        console.print()

        try:
            if current_stage == "features":
                _run_features()
            elif current_stage == "train":
                _run_training(model_type, register)
            elif current_stage == "evaluate":
                _run_evaluation()
            elif current_stage == "monitor":
                _run_monitoring(drift_level)
            elif current_stage == "serve":
                _run_server()
            else:
                console.print(f"[red]Unknown stage: {current_stage}[/red]")
                raise typer.Exit(1)

            console.print(f"\n[green]✅ {current_stage.upper()} complete[/green]\n")

        except Exception as e:
            console.print(f"\n[red]❌ {current_stage.upper()} failed: {e}[/red]\n")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)

    console.print()
    console.rule("[bold green]🏁 Pipeline Complete[/bold green]")


def _run_features() -> None:
    """Generate features and populate the feature store."""
    from feature_store.populate_features import main as populate
    populate()


def _run_training(model_type: str, register: bool) -> None:
    """Train the model with MLflow tracking."""
    from training.train import run_training
    run_training(model_type=model_type, register=register)


def _run_evaluation() -> None:
    """Evaluate and compare model runs."""
    from training.evaluate import compare_runs, load_config
    config = load_config()
    compare_runs(config["mlflow"]["experiment_name"])


def _run_monitoring(drift_level: float) -> None:
    """Run drift monitoring."""
    from monitoring.drift_detector import run_full_monitoring
    run_full_monitoring(drift_level=drift_level)


def _run_server() -> None:
    """Start the FastAPI model server."""
    import uvicorn
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    host = config.get("serving", {}).get("host", "0.0.0.0")
    port = config.get("serving", {}).get("port", 8000)

    console.print(f"  🚀 Starting server at http://{host}:{port}")
    console.print("  📖 API docs at http://{host}:{port}/docs")
    uvicorn.run("serving.app:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    app()
