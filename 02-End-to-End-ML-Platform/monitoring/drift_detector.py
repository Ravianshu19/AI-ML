"""Data and model drift detection using Evidently AI.

Detects feature drift, prediction drift, and data quality issues
by comparing production data against a reference dataset.

Usage:
    python -m monitoring.drift_detector
    python -m monitoring.drift_detector --generate-report --output reports/
"""

from __future__ import annotations

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfRowsWithMissingValues,
    TestNumberOfDuplicatedRows,
    TestColumnsType,
    TestShareOfDriftedColumns,
    TestNumberOfDriftedColumns,
)

console = Console()
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ─── Column Mapping ──────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "age", "tenure_months", "senior_citizen",
    "monthly_logins", "monthly_transactions", "support_tickets_30d",
    "avg_session_duration_min", "days_since_last_login", "product_usage_score",
    "monthly_charges", "total_charges", "late_payments_6m", "has_autopay",
]

column_mapping = ColumnMapping(
    target="churn",
    prediction=None,
    numerical_features=[
        "age", "tenure_months", "monthly_logins", "monthly_transactions",
        "support_tickets_30d", "avg_session_duration_min",
        "days_since_last_login", "product_usage_score",
        "monthly_charges", "total_charges", "late_payments_6m",
    ],
    categorical_features=["senior_citizen", "has_autopay"],
)


def simulate_production_data(
    reference: pd.DataFrame,
    drift_level: float = 0.1,
) -> pd.DataFrame:
    """Simulate production data with configurable drift.

    Introduces controlled drift to test the monitoring pipeline.

    Args:
        reference: Reference dataset to base production data on.
        drift_level: Amount of drift (0.0 = no drift, 1.0 = heavy drift).

    Returns:
        Simulated production DataFrame.
    """
    np.random.seed(99)
    production = reference.copy()
    n = len(production)

    # Introduce drift in numerical features
    for col in ["monthly_charges", "support_tickets_30d", "days_since_last_login"]:
        if col in production.columns:
            noise = np.random.normal(0, drift_level * production[col].std(), n)
            production[col] = production[col] + noise

    # Shift distributions
    if drift_level > 0.3:
        production["monthly_charges"] *= (1 + drift_level)
        production["support_tickets_30d"] += np.random.poisson(2, n)

    # Ensure valid ranges
    production = production.clip(lower=0)

    return production


def run_data_drift_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    output_dir: str | Path = "monitoring/reports",
) -> dict[str, Any]:
    """Generate an Evidently data drift report.

    Args:
        reference: Reference (training) dataset.
        production: Current production dataset.
        output_dir: Directory to save report HTML.

    Returns:
        Dictionary with drift detection results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n📊 Running Data Drift Analysis...")

    # Create drift report
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])

    report.run(
        reference_data=reference[FEATURE_COLUMNS],
        current_data=production[FEATURE_COLUMNS],
        column_mapping=column_mapping,
    )

    # Save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = output_dir / f"data_drift_report_{timestamp}.html"
    report.save_html(str(html_path))
    console.print(f"  💾 Report saved: {html_path}")

    # Extract results
    result = report.as_dict()
    metrics = result.get("metrics", [])

    drift_summary = {
        "timestamp": timestamp,
        "report_path": str(html_path),
        "dataset_drift": False,
        "drifted_columns": [],
        "drift_share": 0.0,
    }

    for metric in metrics:
        metric_result = metric.get("result", {})
        if "dataset_drift" in metric_result:
            drift_summary["dataset_drift"] = metric_result["dataset_drift"]
            drift_summary["drift_share"] = metric_result.get("share_of_drifted_columns", 0)
            drift_summary["n_drifted"] = metric_result.get("number_of_drifted_columns", 0)
            drift_summary["n_columns"] = metric_result.get("number_of_columns", 0)

    return drift_summary


def run_data_quality_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    output_dir: str | Path = "monitoring/reports",
) -> dict[str, Any]:
    """Generate an Evidently data quality report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n🔍 Running Data Quality Analysis...")

    report = Report(metrics=[DataQualityPreset()])
    report.run(
        reference_data=reference[FEATURE_COLUMNS],
        current_data=production[FEATURE_COLUMNS],
        column_mapping=column_mapping,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = output_dir / f"data_quality_report_{timestamp}.html"
    report.save_html(str(html_path))
    console.print(f"  💾 Report saved: {html_path}")

    return {"report_path": str(html_path), "timestamp": timestamp}


def run_test_suite(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    drift_threshold: float = 0.3,
) -> dict[str, Any]:
    """Run Evidently test suite for automated pass/fail checks.

    Args:
        reference: Reference dataset.
        production: Current production dataset.
        drift_threshold: Maximum share of drifted columns allowed.

    Returns:
        Test results summary.
    """
    console.print("\n🧪 Running Automated Test Suite...")

    test_suite = TestSuite(tests=[
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
        TestNumberOfDuplicatedRows(),
        TestColumnsType(),
        TestShareOfDriftedColumns(lt=drift_threshold),
    ])

    test_suite.run(
        reference_data=reference[FEATURE_COLUMNS],
        current_data=production[FEATURE_COLUMNS],
        column_mapping=column_mapping,
    )

    result = test_suite.as_dict()
    tests = result.get("tests", [])

    # Display results
    table = Table(title="🧪 Test Suite Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Description")

    all_passed = True
    for test in tests:
        status = test.get("status", "UNKNOWN")
        passed = status == "SUCCESS"
        if not passed:
            all_passed = False
        table.add_row(
            test.get("name", "Unknown"),
            "[green]✅ PASS[/green]" if passed else "[red]❌ FAIL[/red]",
            test.get("description", ""),
        )

    console.print(table)

    return {
        "all_passed": all_passed,
        "total_tests": len(tests),
        "passed": sum(1 for t in tests if t.get("status") == "SUCCESS"),
        "failed": sum(1 for t in tests if t.get("status") != "SUCCESS"),
    }


def run_full_monitoring(
    drift_level: float = 0.1,
    output_dir: str = "monitoring/reports",
) -> dict[str, Any]:
    """Run the complete monitoring pipeline.

    Args:
        drift_level: Simulated drift level (0.0 to 1.0).
        output_dir: Directory for report output.

    Returns:
        Combined monitoring results.
    """
    config = load_config()
    threshold = config.get("monitoring", {}).get("drift_threshold", 0.05)

    console.print(
        Panel(
            f"[bold cyan]📡 Drift Monitoring Pipeline[/bold cyan]\n\n"
            f"Drift level: {drift_level}\n"
            f"Threshold: {threshold}\n"
            f"Output: {output_dir}",
            border_style="cyan",
        )
    )

    # Load reference data
    ref_path = Path("data/processed/reference.csv")
    if not ref_path.exists():
        console.print("[red]Reference data not found. Run feature population first.[/red]")
        return {}

    reference = pd.read_csv(ref_path)
    console.print(f"  📋 Reference data: {len(reference)} records")

    # Simulate production data
    production = simulate_production_data(reference, drift_level=drift_level)
    console.print(f"  📋 Production data: {len(production)} records (simulated)")

    # Run analyses
    drift_results = run_data_drift_report(reference, production, output_dir)
    quality_results = run_data_quality_report(reference, production, output_dir)
    test_results = run_test_suite(reference, production, threshold)

    # Summary
    console.print()
    is_drifted = drift_results.get("dataset_drift", False)
    status_color = "red" if is_drifted else "green"
    status_icon = "🚨" if is_drifted else "✅"

    console.print(
        Panel(
            f"[bold {status_color}]{status_icon} Monitoring Summary[/bold {status_color}]\n\n"
            f"Dataset drift detected: {is_drifted}\n"
            f"Drifted columns: {drift_results.get('n_drifted', 0)}/{drift_results.get('n_columns', 0)}\n"
            f"Drift share: {drift_results.get('drift_share', 0):.1%}\n"
            f"Tests passed: {test_results.get('passed', 0)}/{test_results.get('total_tests', 0)}",
            border_style=status_color,
        )
    )

    return {
        "drift": drift_results,
        "quality": quality_results,
        "tests": test_results,
        "alert": is_drifted,
    }


if __name__ == "__main__":
    import typer

    def main(
        drift_level: float = typer.Option(0.1, "--drift-level", "-d", help="Simulated drift level"),
        output: str = typer.Option("monitoring/reports", "--output", "-o", help="Report output directory"),
    ):
        run_full_monitoring(drift_level=drift_level, output_dir=output)

    typer.run(main)
