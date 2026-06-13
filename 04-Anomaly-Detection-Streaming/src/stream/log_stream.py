"""Simulated log/event data stream with anomalous patterns.

Generates realistic application log events with:
- Normal operational patterns (varying by service)
- Anomalous bursts, unusual error rates, latency spikes
- Seasonal patterns (higher traffic during "business hours")

Usage:
    from src.stream.log_stream import LogStream

    stream = LogStream()
    for event in stream.generate():
        print(event)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator

import numpy as np


@dataclass
class LogEvent:
    """A single log/event entry."""
    timestamp: str
    service: str
    level: str            # INFO, WARN, ERROR, CRITICAL
    status_code: int
    response_time_ms: float
    request_count: int
    error_rate: float
    cpu_usage: float
    memory_usage_mb: float
    is_anomaly: bool = False
    anomaly_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "service": self.service,
            "level": self.level,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
            "request_count": self.request_count,
            "error_rate": self.error_rate,
            "cpu_usage": self.cpu_usage,
            "memory_usage_mb": self.memory_usage_mb,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
        }

    def features(self) -> dict[str, float]:
        """Return numeric features for the detector."""
        return {
            "response_time_ms": self.response_time_ms,
            "request_count": float(self.request_count),
            "error_rate": self.error_rate,
            "cpu_usage": self.cpu_usage,
            "memory_usage_mb": self.memory_usage_mb,
        }


# Service profiles: baseline behavior per service
SERVICE_PROFILES = {
    "api-gateway": {
        "base_response_ms": 45,
        "base_requests": 100,
        "base_error_rate": 0.005,
        "base_cpu": 35.0,
        "base_memory_mb": 512,
    },
    "auth-service": {
        "base_response_ms": 30,
        "base_requests": 50,
        "base_error_rate": 0.002,
        "base_cpu": 20.0,
        "base_memory_mb": 256,
    },
    "payment-service": {
        "base_response_ms": 120,
        "base_requests": 30,
        "base_error_rate": 0.008,
        "base_cpu": 40.0,
        "base_memory_mb": 768,
    },
    "user-service": {
        "base_response_ms": 25,
        "base_requests": 80,
        "base_error_rate": 0.003,
        "base_cpu": 25.0,
        "base_memory_mb": 384,
    },
    "notification-service": {
        "base_response_ms": 60,
        "base_requests": 40,
        "base_error_rate": 0.01,
        "base_cpu": 15.0,
        "base_memory_mb": 192,
    },
}


class LogStream:
    """Simulated application log/event stream.

    Generates realistic microservice log events with configurable
    anomaly injection for testing detection algorithms.

    Args:
        services: List of service names (uses defaults if None).
        anomaly_prob: Probability of anomaly per event.
        seed: Random seed.
    """

    def __init__(
        self,
        services: list[str] | None = None,
        anomaly_prob: float = 0.03,
        seed: int = 42,
    ):
        self.services = services or list(SERVICE_PROFILES.keys())
        self.anomaly_prob = anomaly_prob
        self.rng = np.random.default_rng(seed)
        self._step = 0

    def _generate_normal_event(self, service: str) -> dict[str, Any]:
        """Generate a normal log event for a service."""
        profile = SERVICE_PROFILES.get(service, SERVICE_PROFILES["api-gateway"])

        # Add time-based variation (simulating daily traffic patterns)
        t = self._step * 0.01
        traffic_mult = 1.0 + 0.3 * math.sin(2 * math.pi * t / 8640)

        response_ms = max(5, profile["base_response_ms"] * traffic_mult
                          + self.rng.normal(0, profile["base_response_ms"] * 0.15))

        requests = max(1, int(profile["base_requests"] * traffic_mult
                              + self.rng.normal(0, 5)))

        error_rate = max(0, profile["base_error_rate"]
                         + self.rng.normal(0, 0.002))

        cpu = max(0, min(100, profile["base_cpu"] * traffic_mult
                         + self.rng.normal(0, 3)))

        memory = max(50, profile["base_memory_mb"]
                     + self.rng.normal(0, 20))

        # Determine log level based on error rate
        if error_rate > 0.05:
            level = "ERROR"
            status_code = self.rng.choice([500, 502, 503])
        elif error_rate > 0.02:
            level = "WARN"
            status_code = self.rng.choice([200, 408, 429])
        else:
            level = "INFO"
            status_code = 200

        return {
            "response_time_ms": round(response_ms, 2),
            "request_count": requests,
            "error_rate": round(error_rate, 4),
            "cpu_usage": round(cpu, 1),
            "memory_usage_mb": round(memory, 1),
            "level": level,
            "status_code": int(status_code),
        }

    def _inject_anomaly(self, event: dict) -> tuple[dict, str]:
        """Inject an anomaly into the event."""
        anomaly_type = self.rng.choice(
            ["latency_spike", "error_burst", "traffic_surge",
             "memory_leak", "cpu_spike", "cascade_failure"],
            p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10],
        )

        if anomaly_type == "latency_spike":
            event["response_time_ms"] *= self.rng.uniform(5, 20)
            event["level"] = "WARN"
            event["status_code"] = 408

        elif anomaly_type == "error_burst":
            event["error_rate"] = self.rng.uniform(0.15, 0.60)
            event["level"] = "ERROR"
            event["status_code"] = self.rng.choice([500, 502, 503])

        elif anomaly_type == "traffic_surge":
            event["request_count"] = int(event["request_count"] * self.rng.uniform(5, 15))
            event["cpu_usage"] = min(99, event["cpu_usage"] * 2)

        elif anomaly_type == "memory_leak":
            event["memory_usage_mb"] *= self.rng.uniform(2, 5)
            event["cpu_usage"] = min(99, event["cpu_usage"] * 1.5)

        elif anomaly_type == "cpu_spike":
            event["cpu_usage"] = self.rng.uniform(85, 99)
            event["response_time_ms"] *= self.rng.uniform(2, 5)

        elif anomaly_type == "cascade_failure":
            event["error_rate"] = self.rng.uniform(0.3, 0.8)
            event["response_time_ms"] *= self.rng.uniform(10, 50)
            event["cpu_usage"] = self.rng.uniform(90, 99)
            event["level"] = "CRITICAL"
            event["status_code"] = 503

        return event, anomaly_type

    def generate(self, max_samples: int | None = None) -> Generator[LogEvent, None, None]:
        """Generate a continuous stream of log events.

        Yields:
            LogEvent objects.
        """
        count = 0
        while max_samples is None or count < max_samples:
            for service in self.services:
                event_data = self._generate_normal_event(service)

                is_anomaly = False
                anomaly_type = None

                if self.rng.random() < self.anomaly_prob:
                    event_data, anomaly_type = self._inject_anomaly(event_data)
                    is_anomaly = True

                yield LogEvent(
                    timestamp=datetime.now().isoformat(),
                    service=service,
                    is_anomaly=is_anomaly,
                    anomaly_type=anomaly_type,
                    **event_data,
                )

                count += 1
                if max_samples and count >= max_samples:
                    return

            self._step += 1
