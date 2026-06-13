"""Real-time streaming anomaly detection pipeline.

Connects stream sources to online detectors and routes alerts.
Processes data point-by-point with O(1) memory per detector.

Usage:
    from src.pipeline.streaming_pipeline import StreamingPipeline

    pipeline = StreamingPipeline.from_config("config.yaml")
    pipeline.run(source="sensor", max_samples=10000)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

from src.stream.sensor_stream import SensorStream
from src.stream.log_stream import LogStream
from src.detectors.online_detectors import (
    EnsembleDetector,
    DetectionResult,
    build_detector_from_config,
)

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Running statistics for the pipeline."""
    total_processed: int = 0
    total_anomalies_detected: int = 0
    total_true_anomalies: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def precision(self) -> float:
        tp_fp = self.true_positives + self.false_positives
        return self.true_positives / tp_fp if tp_fp > 0 else 0.0

    @property
    def recall(self) -> float:
        tp_fn = self.true_positives + self.false_negatives
        return self.true_positives / tp_fn if tp_fn > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def throughput(self) -> float:
        elapsed = time.time() - self.start_time
        return self.total_processed / elapsed if elapsed > 0 else 0.0


class AlertManager:
    """Manages anomaly alerts with cooldown to prevent alert fatigue.

    Args:
        cooldown_seconds: Minimum seconds between alerts for the same source.
    """

    def __init__(self, cooldown_seconds: float = 30.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_alert: dict[str, float] = {}
        self.alert_history: list[dict[str, Any]] = []

    def should_alert(self, source_id: str) -> bool:
        now = time.time()
        last = self._last_alert.get(source_id, 0)
        return (now - last) >= self.cooldown_seconds

    def record_alert(
        self,
        source_id: str,
        score: float,
        details: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Record an alert if cooldown has passed.

        Returns:
            Alert dict if emitted, None if suppressed.
        """
        if not self.should_alert(source_id):
            return None

        self._last_alert[source_id] = time.time()

        severity = (
            "🔴 CRITICAL" if score > 0.9
            else "🟠 HIGH" if score > 0.7
            else "🟡 MEDIUM" if score > 0.5
            else "🟢 LOW"
        )

        alert = {
            "timestamp": datetime.now().isoformat(),
            "source_id": source_id,
            "severity": severity,
            "anomaly_score": round(score, 4),
            "details": details,
        }

        self.alert_history.append(alert)
        return alert


class StreamingPipeline:
    """Main streaming anomaly detection pipeline.

    Orchestrates:
    1. Data ingestion from stream sources
    2. Feature extraction
    3. Online anomaly detection (learn + score)
    4. Alert management
    5. Statistics tracking

    Args:
        detector: Configured EnsembleDetector.
        alert_manager: AlertManager instance.
        detection_threshold: Score threshold for flagging anomalies.
    """

    def __init__(
        self,
        detector: EnsembleDetector,
        alert_manager: AlertManager,
        detection_threshold: float = 0.5,
    ):
        self.detector = detector
        self.alert_manager = alert_manager
        self.threshold = detection_threshold
        self.stats = PipelineStats()
        self.recent_anomalies: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "StreamingPipeline":
        """Build pipeline from config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        detector = build_detector_from_config(config.get("detectors", {}))
        alert_config = config.get("alerts", {})
        alert_manager = AlertManager(
            cooldown_seconds=alert_config.get("cooldown_seconds", 30),
        )

        return cls(
            detector=detector,
            alert_manager=alert_manager,
        )

    def process_one(
        self,
        features: dict[str, float],
        source_id: str,
        ground_truth: bool = False,
    ) -> DetectionResult:
        """Process a single data point through the pipeline.

        This is the core loop:
        1. Score the observation (before learning)
        2. Learn from the observation (update model)
        3. Track statistics
        4. Emit alerts if needed

        Args:
            features: Numeric feature dictionary.
            source_id: Identifier for the data source.
            ground_truth: Whether this is actually an anomaly.

        Returns:
            DetectionResult with score and classification.
        """
        # 1. Detect (score before learning for proper evaluation)
        result = self.detector.detect(features, self.threshold)

        # 2. Learn (update model incrementally)
        self.detector.learn_one(features)

        # 3. Track stats
        self.stats.total_processed += 1
        if ground_truth:
            self.stats.total_true_anomalies += 1

        if result.is_anomaly:
            self.stats.total_anomalies_detected += 1

            if ground_truth:
                self.stats.true_positives += 1
            else:
                self.stats.false_positives += 1

            # Emit alert
            alert = self.alert_manager.record_alert(
                source_id=source_id,
                score=result.anomaly_score,
                details=result.details,
            )
            if alert:
                self.recent_anomalies.append(alert)
                # Keep only last 50
                if len(self.recent_anomalies) > 50:
                    self.recent_anomalies = self.recent_anomalies[-50:]

        elif ground_truth:
            self.stats.false_negatives += 1

        return result

    def _build_stats_table(self) -> Table:
        """Build a rich table with pipeline statistics."""
        table = Table(title="📊 Pipeline Statistics", show_lines=True)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="green", justify="right", width=15)

        s = self.stats
        table.add_row("Total Processed", f"{s.total_processed:,}")
        table.add_row("Anomalies Detected", f"{s.total_anomalies_detected:,}")
        table.add_row("True Anomalies (GT)", f"{s.total_true_anomalies:,}")
        table.add_row("True Positives", f"{s.true_positives:,}")
        table.add_row("False Positives", f"{s.false_positives:,}")
        table.add_row("False Negatives", f"{s.false_negatives:,}")
        table.add_row("─" * 20, "─" * 10)
        table.add_row("Precision", f"{s.precision:.4f}")
        table.add_row("Recall", f"{s.recall:.4f}")
        table.add_row("F1 Score", f"{s.f1:.4f}")
        table.add_row("Throughput", f"{s.throughput:.0f} pts/sec")

        return table

    def _build_alerts_table(self) -> Table:
        """Build a table showing recent alerts."""
        table = Table(title="🚨 Recent Alerts", show_lines=True)
        table.add_column("Time", style="dim", width=12)
        table.add_column("Source", style="cyan", width=20)
        table.add_column("Severity", width=15)
        table.add_column("Score", justify="right", width=8)

        for alert in self.recent_anomalies[-10:]:
            ts = alert["timestamp"].split("T")[1][:8]
            table.add_row(
                ts,
                alert["source_id"],
                alert["severity"],
                f"{alert['anomaly_score']:.3f}",
            )

        return table

    def run_sensor_stream(
        self,
        max_samples: int = 5000,
        show_live: bool = True,
        config: dict | None = None,
    ) -> PipelineStats:
        """Run the pipeline on simulated sensor data."""
        sensor_config = (config or {}).get("stream", {}).get("sensor", {})
        stream = SensorStream(
            num_sensors=sensor_config.get("num_sensors", 5),
            anomaly_prob=sensor_config.get("anomaly_probability", 0.02),
            anomaly_magnitude=sensor_config.get("anomaly_magnitude", 5.0),
            drift_rate=sensor_config.get("drift_rate", 0.001),
        )

        console.print(Panel(
            f"[bold cyan]📡 Sensor Stream — Online Anomaly Detection[/bold cyan]\n\n"
            f"Samples: {max_samples:,}\n"
            f"Detectors: {', '.join(d.name for d in self.detector.detectors)}\n"
            f"Ensemble: {self.detector.method}",
            border_style="cyan",
        ))

        if show_live:
            with Live(console=console, refresh_per_second=4) as live:
                for reading in stream.generate(max_samples=max_samples):
                    self.process_one(
                        features=reading.features(),
                        source_id=reading.sensor_id,
                        ground_truth=reading.is_anomaly,
                    )

                    if self.stats.total_processed % 100 == 0:
                        from rich.columns import Columns
                        live.update(Columns([
                            self._build_stats_table(),
                            self._build_alerts_table(),
                        ]))
        else:
            for reading in stream.generate(max_samples=max_samples):
                self.process_one(
                    features=reading.features(),
                    source_id=reading.sensor_id,
                    ground_truth=reading.is_anomaly,
                )

        # Final stats
        console.print()
        console.print(self._build_stats_table())
        console.print(self._build_alerts_table())

        return self.stats

    def run_log_stream(
        self,
        max_samples: int = 5000,
        show_live: bool = True,
        config: dict | None = None,
    ) -> PipelineStats:
        """Run the pipeline on simulated log data."""
        log_config = (config or {}).get("stream", {}).get("log", {})
        stream = LogStream(
            services=log_config.get("services"),
            anomaly_prob=log_config.get("anomaly_probability", 0.03),
        )

        console.print(Panel(
            f"[bold magenta]📋 Log Stream — Online Anomaly Detection[/bold magenta]\n\n"
            f"Samples: {max_samples:,}\n"
            f"Services: {', '.join(stream.services)}\n"
            f"Detectors: {', '.join(d.name for d in self.detector.detectors)}",
            border_style="magenta",
        ))

        if show_live:
            with Live(console=console, refresh_per_second=4) as live:
                for event in stream.generate(max_samples=max_samples):
                    self.process_one(
                        features=event.features(),
                        source_id=event.service,
                        ground_truth=event.is_anomaly,
                    )

                    if self.stats.total_processed % 100 == 0:
                        from rich.columns import Columns
                        live.update(Columns([
                            self._build_stats_table(),
                            self._build_alerts_table(),
                        ]))
        else:
            for event in stream.generate(max_samples=max_samples):
                self.process_one(
                    features=event.features(),
                    source_id=event.service,
                    ground_truth=event.is_anomaly,
                )

        console.print()
        console.print(self._build_stats_table())
        console.print(self._build_alerts_table())

        return self.stats
