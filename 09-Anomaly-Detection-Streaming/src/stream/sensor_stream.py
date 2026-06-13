"""Simulated sensor data stream with realistic patterns.

Generates continuous time-series data from virtual IoT sensors with:
- Sinusoidal base signals (daily/hourly cycles)
- Gaussian noise
- Injected anomalies (point, contextual, collective)
- Gradual concept drift

Usage:
    from src.stream.sensor_stream import SensorStream

    stream = SensorStream(num_sensors=5)
    for reading in stream.generate():
        print(reading)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generator, Any

import numpy as np


@dataclass
class SensorReading:
    """A single sensor reading."""
    timestamp: str
    sensor_id: str
    value: float
    temperature: float
    pressure: float
    vibration: float
    is_anomaly: bool = False       # Ground truth label
    anomaly_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "sensor_id": self.sensor_id,
            "value": self.value,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "vibration": self.vibration,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
        }

    def features(self) -> dict[str, float]:
        """Return numeric features for the detector."""
        return {
            "value": self.value,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "vibration": self.vibration,
        }


class SensorStream:
    """Simulated multi-sensor data stream with anomaly injection.

    Generates realistic sensor data with configurable anomaly patterns:
    - Point anomalies: sudden spikes
    - Contextual anomalies: normal values at wrong times
    - Collective anomalies: sustained unusual patterns
    - Concept drift: gradual distribution shift

    Args:
        num_sensors: Number of virtual sensors.
        noise_std: Standard deviation of Gaussian noise.
        anomaly_prob: Probability of anomaly per reading.
        anomaly_magnitude: Magnitude of injected anomalies (in std devs).
        drift_rate: Rate of concept drift per sample.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_sensors: int = 5,
        noise_std: float = 0.5,
        anomaly_prob: float = 0.02,
        anomaly_magnitude: float = 5.0,
        drift_rate: float = 0.001,
        seed: int = 42,
    ):
        self.num_sensors = num_sensors
        self.noise_std = noise_std
        self.anomaly_prob = anomaly_prob
        self.anomaly_magnitude = anomaly_magnitude
        self.drift_rate = drift_rate
        self.rng = np.random.default_rng(seed)
        self._step = 0
        self._drift_offset = 0.0

        # Per-sensor baselines
        self._baselines = {
            f"sensor_{i:02d}": {
                "value_base": 50 + self.rng.uniform(-10, 10),
                "temp_base": 22 + self.rng.uniform(-3, 3),
                "pressure_base": 1013 + self.rng.uniform(-5, 5),
                "vibration_base": 0.5 + self.rng.uniform(0, 0.3),
                "phase_offset": self.rng.uniform(0, 2 * math.pi),
            }
            for i in range(num_sensors)
        }

    def _generate_base_signal(self, step: int, baseline: dict) -> dict[str, float]:
        """Generate a base signal with sinusoidal patterns."""
        t = step * 0.01  # Time progression

        # Sinusoidal patterns (simulating daily/hourly cycles)
        hourly_cycle = math.sin(2 * math.pi * t / 360 + baseline["phase_offset"])
        daily_cycle = math.sin(2 * math.pi * t / 8640 + baseline["phase_offset"])

        value = (
            baseline["value_base"]
            + 5 * hourly_cycle
            + 2 * daily_cycle
            + self._drift_offset
            + self.rng.normal(0, self.noise_std)
        )

        temperature = (
            baseline["temp_base"]
            + 2 * daily_cycle
            + self.rng.normal(0, 0.3)
        )

        pressure = (
            baseline["pressure_base"]
            + 1.5 * hourly_cycle
            + self.rng.normal(0, 0.2)
        )

        vibration = abs(
            baseline["vibration_base"]
            + 0.1 * hourly_cycle
            + self.rng.normal(0, 0.05)
        )

        return {
            "value": round(value, 3),
            "temperature": round(temperature, 2),
            "pressure": round(pressure, 2),
            "vibration": round(vibration, 4),
        }

    def _inject_anomaly(self, signals: dict[str, float]) -> tuple[dict[str, float], str]:
        """Inject an anomaly into the signal."""
        anomaly_type = self.rng.choice(
            ["point_spike", "point_drop", "multi_feature", "frozen"],
            p=[0.35, 0.25, 0.25, 0.15],
        )

        if anomaly_type == "point_spike":
            feature = self.rng.choice(["value", "temperature", "vibration"])
            signals[feature] += self.anomaly_magnitude * self.noise_std * self.rng.uniform(2, 5)

        elif anomaly_type == "point_drop":
            feature = self.rng.choice(["value", "pressure"])
            signals[feature] -= self.anomaly_magnitude * self.noise_std * self.rng.uniform(2, 4)

        elif anomaly_type == "multi_feature":
            # Multiple features shift simultaneously
            signals["value"] += self.anomaly_magnitude * self.rng.uniform(1, 3)
            signals["vibration"] *= self.rng.uniform(3, 8)
            signals["temperature"] += self.rng.uniform(3, 8)

        elif anomaly_type == "frozen":
            # All readings become identical (sensor stuck)
            frozen_val = signals["value"]
            signals = {k: frozen_val for k in signals}

        return signals, anomaly_type

    def generate(self, max_samples: int | None = None) -> Generator[SensorReading, None, None]:
        """Generate a continuous stream of sensor readings.

        Args:
            max_samples: Maximum number of samples (None = infinite).

        Yields:
            SensorReading objects.
        """
        count = 0
        while max_samples is None or count < max_samples:
            for sensor_id, baseline in self._baselines.items():
                signals = self._generate_base_signal(self._step, baseline)

                is_anomaly = False
                anomaly_type = None

                if self.rng.random() < self.anomaly_prob:
                    signals, anomaly_type = self._inject_anomaly(signals)
                    is_anomaly = True

                yield SensorReading(
                    timestamp=datetime.now().isoformat(),
                    sensor_id=sensor_id,
                    is_anomaly=is_anomaly,
                    anomaly_type=anomaly_type,
                    **signals,
                )

                count += 1
                if max_samples and count >= max_samples:
                    return

            # Apply concept drift
            self._drift_offset += self.drift_rate
            self._step += 1

    def generate_batch(self, n: int) -> list[dict[str, Any]]:
        """Generate a batch of readings as dictionaries."""
        return [r.to_dict() for r in self.generate(max_samples=n)]
