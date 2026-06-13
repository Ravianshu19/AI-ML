"""Model evaluation and comparison utilities.

Compare multiple registered models from MLflow and select
the best candidate for promotion to Production.

Usage:
    python -m training.evaluate
    python -m training.evaluate --promote-best
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def compare_runs(experiment_name: str, top_n: int = 5) -> pd.DataFrame:
    """Compare the top N runs in an experiment by F1 score.

    Returns:
        DataFrame with run details and metrics.
    """
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        console.print(f"[red]Experiment '{experiment_name}' not found.[/red]")
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=top_n,
    )

    if runs.empty:
        console.print("[yellow]No runs found.[/yellow]")
        return runs

    # Display comparison table
    table = Table(title=f"🏆 Top {top_n} Runs — {experiment_name}")
    table.add_column("Run ID", style="dim")
    table.add_column("Model Type", style="cyan")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right", style="bold green")
    table.add_column("ROC AUC", justify="right")

    for _, run in runs.iterrows():
        table.add_row(
            run["run_id"][:8],
            str(run.get("params.model_type", "N/A")),
            f"{run.get('metrics.accuracy', 0):.4f}",
            f"{run.get('metrics.precision', 0):.4f}",
            f"{run.get('metrics.recall', 0):.4f}",
            f"{run.get('metrics.f1', 0):.4f}",
            f"{run.get('metrics.roc_auc', 0):.4f}",
        )

    console.print(table)
    return runs


def promote_best_model(
    experiment_name: str,
    model_name: str,
    metric: str = "f1",
) -> str | None:
    """Find the best run and promote its model to Production.

    Args:
        experiment_name: MLflow experiment name.
        model_name: Registered model name.
        metric: Metric to optimize for.

    Returns:
        The run_id of the promoted model, or None.
    """
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    # Find best run
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        console.print(f"[red]Experiment '{experiment_name}' not found.[/red]")
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if runs.empty:
        console.print("[red]No runs found to promote.[/red]")
        return None

    best_run = runs.iloc[0]
    best_run_id = best_run["run_id"]
    best_metric = best_run.get(f"metrics.{metric}", 0)

    console.print(
        f"\n🏆 Best run: {best_run_id[:8]} "
        f"({metric}={best_metric:.4f})"
    )

    # Register if not already registered
    model_uri = f"runs:/{best_run_id}/model"
    try:
        mv = mlflow.register_model(model_uri, model_name)
        console.print(f"  ✅ Registered as {model_name} v{mv.version}")

        # Transition to Production
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        console.print(f"  🚀 Promoted v{mv.version} to Production")
        return best_run_id

    except Exception as e:
        console.print(f"  [red]Error promoting model: {e}[/red]")
        return None


if __name__ == "__main__":
    import typer

    def main(
        promote_best: bool = typer.Option(False, "--promote-best", help="Promote best model to Production"),
    ):
        config = load_config()
        experiment = config["mlflow"]["experiment_name"]
        model_name = config["mlflow"]["model_name"]

        compare_runs(experiment)

        if promote_best:
            promote_best_model(experiment, model_name)

    typer.run(main)
