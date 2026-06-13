"""Model training pipeline with MLflow tracking.

Trains a churn prediction model, logs metrics/params/artifacts
to MLflow, and optionally registers the model in the registry.

Usage:
    python -m training.train
    python -m training.train --model-type gradient_boosting --register
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

console = Console()
logger = logging.getLogger(__name__)

# ─── Model Registry ─────────────────────────────────────────────────
MODEL_CLASSES = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}


def load_config() -> dict:
    """Load the pipeline configuration."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Load and prepare training data.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    data_path = Path("data/processed/training_data.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            "Run: python -m feature_store.populate_features"
        )

    df = pd.read_csv(data_path)
    console.print(f"  📊 Loaded {len(df)} records, {df['churn'].mean():.1%} churn rate")

    # Feature columns (exclude customer_id and target)
    feature_cols = [
        "age", "tenure_months", "senior_citizen",
        "monthly_logins", "monthly_transactions", "support_tickets_30d",
        "avg_session_duration_min", "days_since_last_login", "product_usage_score",
        "monthly_charges", "total_charges", "late_payments_6m", "has_autopay",
    ]

    X = df[feature_cols].copy()
    y = df["churn"].copy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    # Scale features
    scaler = StandardScaler()
    X_train[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test[feature_cols] = scaler.transform(X_test[feature_cols])

    console.print(f"  📊 Train: {len(X_train)}, Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test, scaler


def train_model(
    model_type: str,
    hyperparams: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Train the specified model."""
    model_class = MODEL_CLASSES.get(model_type)
    if model_class is None:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: {list(MODEL_CLASSES.keys())}"
        )

    console.print(f"  🏋️ Training {model_type}...")
    model = model_class(**hyperparams)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics


def display_metrics(metrics: dict[str, float], model_type: str) -> None:
    """Display metrics in a rich table."""
    table = Table(title=f"📈 {model_type} — Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for name, value in metrics.items():
        table.add_row(name.upper(), f"{value:.4f}")

    console.print(table)


def run_training(
    model_type: str | None = None,
    register: bool = False,
) -> dict[str, Any]:
    """Execute the full training pipeline with MLflow tracking.

    Args:
        model_type: Type of model to train. Uses config default if None.
        register: Whether to register the model in MLflow registry.

    Returns:
        Dictionary with model, metrics, and run info.
    """
    config = load_config()
    model_type = model_type or config["training"]["model_type"]
    hyperparams = config["training"]["hyperparameters"][model_type]

    console.print()
    console.print(
        Panel(
            f"[bold cyan]🚀 Training Pipeline[/bold cyan]\n\n"
            f"Model: {model_type}\n"
            f"MLflow: {config['mlflow']['tracking_uri']}\n"
            f"Experiment: {config['mlflow']['experiment_name']}",
            border_style="cyan",
        )
    )

    # Setup MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(config)

    # MLflow run
    with mlflow.start_run(run_name=f"{model_type}_run") as run:
        # Log parameters
        mlflow.log_params(hyperparams)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])

        # Train
        model = train_model(model_type, hyperparams, X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        display_metrics(metrics, model_type)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=config["mlflow"]["artifact_path"],
            registered_model_name=(
                config["mlflow"]["model_name"] if register else None
            ),
        )

        # Save and log scaler
        scaler_path = Path("artifacts/scaler.joblib")
        scaler_path.parent.mkdir(exist_ok=True)
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path))

        # Log feature importance (if available)
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)

            importance_path = Path("artifacts/feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path))

            console.print("\n📊 Top 5 Features:")
            for _, row in importance_df.head(5).iterrows():
                bar = "█" * int(row["importance"] * 50)
                console.print(f"  {row['feature']:30s} {bar} {row['importance']:.4f}")

        console.print(f"\n  ✅ MLflow Run ID: {run.info.run_id}")
        if register:
            console.print(f"  ✅ Model registered as: {config['mlflow']['model_name']}")

    return {
        "model": model,
        "metrics": metrics,
        "run_id": run.info.run_id,
        "scaler": scaler,
    }


# ─── CLI Entry Point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import typer

    def main(
        model_type: str = typer.Option(None, "--model-type", "-m", help="Model type to train"),
        register: bool = typer.Option(False, "--register", "-r", help="Register model in MLflow"),
    ):
        run_training(model_type=model_type, register=register)

    typer.run(main)
