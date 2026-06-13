"""CLI entry point for the streaming anomaly detection system.

Usage:
    python -m src.main sensor --samples 5000
    python -m src.main logs --samples 3000 --anomaly-prob 0.05
    python -m src.main benchmark
    python -m src.main dashboard
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

console = Console()
app = typer.Typer(
    name="anomaly-detect",
    help="🔍 Streaming Anomaly Detection with Online Learning",
    add_completion=False,
)


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False)],
    )


def _load_config() -> dict:
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# ─── Sensor Stream Command ───────────────────────────────────────────
@app.command()
def sensor(
    samples: int = typer.Option(5000, "--samples", "-n", help="Number of samples to process"),
    anomaly_prob: float = typer.Option(0.02, "--anomaly-prob", "-a", help="Anomaly injection probability"),
    live: bool = typer.Option(True, "--live/--no-live", help="Show live dashboard"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run anomaly detection on simulated sensor data stream."""
    _setup_logging(verbose)
    config = _load_config()

    # Override config with CLI args
    config.setdefault("stream", {}).setdefault("sensor", {})
    config["stream"]["sensor"]["anomaly_probability"] = anomaly_prob

    from src.pipeline.streaming_pipeline import StreamingPipeline
    pipeline = StreamingPipeline.from_config()
    pipeline.run_sensor_stream(max_samples=samples, show_live=live, config=config)


# ─── Log Stream Command ──────────────────────────────────────────────
@app.command()
def logs(
    samples: int = typer.Option(5000, "--samples", "-n", help="Number of samples to process"),
    anomaly_prob: float = typer.Option(0.03, "--anomaly-prob", "-a", help="Anomaly injection probability"),
    live: bool = typer.Option(True, "--live/--no-live", help="Show live dashboard"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run anomaly detection on simulated log/event data stream."""
    _setup_logging(verbose)
    config = _load_config()

    config.setdefault("stream", {}).setdefault("log", {})
    config["stream"]["log"]["anomaly_probability"] = anomaly_prob

    from src.pipeline.streaming_pipeline import StreamingPipeline
    pipeline = StreamingPipeline.from_config()
    pipeline.run_log_stream(max_samples=samples, show_live=live, config=config)


# ─── Benchmark Command ───────────────────────────────────────────────
@app.command()
def benchmark(
    samples: int = typer.Option(10000, "--samples", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Benchmark all detectors individually and as an ensemble.

    Runs each detector separately on the same data stream and
    compares their precision, recall, F1, and throughput.
    """
    _setup_logging(verbose)

    from src.stream.sensor_stream import SensorStream
    from src.detectors.online_detectors import (
        HalfSpaceTreesDetector,
        EWMADetector,
        RollingZScoreDetector,
        EnsembleDetector,
    )

    console.print(Panel(
        f"[bold cyan]🏎️  Detector Benchmark[/bold cyan]\n\n"
        f"Samples: {samples:,}\n"
        f"Detectors: HalfSpaceTrees, EWMA, RollingZScore, Ensemble",
        border_style="cyan",
    ))

    # Generate data once
    stream = SensorStream(anomaly_prob=0.03)
    data = [(r.features(), r.is_anomaly, r.sensor_id) for r in stream.generate(max_samples=samples)]

    detectors = [
        HalfSpaceTreesDetector(n_trees=25, height=8, window_size=250),
        EWMADetector(alpha=0.05, threshold_sigmas=3.0, warmup=100),
        RollingZScoreDetector(window_size=200, threshold=3.0),
    ]

    # Add ensemble
    ensemble_detectors = [
        HalfSpaceTreesDetector(n_trees=25, height=8, window_size=250),
        EWMADetector(alpha=0.05, threshold_sigmas=3.0, warmup=100),
        RollingZScoreDetector(window_size=200, threshold=3.0),
    ]
    detectors.append(EnsembleDetector(ensemble_detectors, method="majority_vote", min_agree=2))

    # Benchmark each
    results_table = Table(title="🏆 Benchmark Results")
    results_table.add_column("Detector", style="cyan", width=22)
    results_table.add_column("TP", justify="right")
    results_table.add_column("FP", justify="right")
    results_table.add_column("FN", justify="right")
    results_table.add_column("Precision", justify="right", style="green")
    results_table.add_column("Recall", justify="right", style="green")
    results_table.add_column("F1", justify="right", style="bold green")
    results_table.add_column("Time (s)", justify="right", style="dim")

    for detector in detectors:
        tp = fp = fn = 0
        start = time.time()

        for features, is_anomaly, _ in data:
            result = detector.detect(features, threshold=0.5)
            detector.learn_one(features)

            if result.is_anomaly and is_anomaly:
                tp += 1
            elif result.is_anomaly and not is_anomaly:
                fp += 1
            elif not result.is_anomaly and is_anomaly:
                fn += 1

        elapsed = time.time() - start
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results_table.add_row(
            detector.name,
            str(tp), str(fp), str(fn),
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{elapsed:.2f}",
        )

    console.print()
    console.print(results_table)


# ─── Dashboard Command ───────────────────────────────────────────────
@app.command()
def dashboard() -> None:
    """Launch the Streamlit real-time visualization dashboard."""
    import subprocess
    import sys

    dashboard_path = Path("src/dashboard/app.py")
    if not dashboard_path.exists():
        console.print("[red]Dashboard file not found.[/red]")
        raise typer.Exit(1)

    console.print("[cyan]🚀 Launching Streamlit dashboard...[/cyan]")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


if __name__ == "__main__":
    app()
