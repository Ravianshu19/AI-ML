"""Online anomaly detectors using River and custom algorithms.

Implements multiple online (incremental) anomaly detection algorithms
that learn from a single pass over streaming data without storing
the full dataset in memory.

Detectors:
    1. HalfSpaceTreesDetector — River's Half-Space Trees
    2. EWMADetector — Exponentially Weighted Moving Average
    3. RollingZScoreDetector — Sliding-window Z-score
    4. EnsembleDetector — Combines multiple detectors

All detectors implement the OnlineDetector protocol:
    - learn_one(features: dict) -> None
    - score_one(features: dict) -> float  (higher = more anomalous)
    - is_anomaly(features: dict) -> bool
"""

from __future__ import annotations

import math
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─── Base Protocol ────────────────────────────────────────────────────
@dataclass
class DetectionResult:
    """Result from an anomaly detector."""
    is_anomaly: bool
    anomaly_score: float          # 0.0 = normal, 1.0 = definitely anomalous
    detector_name: str
    details: dict[str, Any] = field(default_factory=dict)


class OnlineDetector(ABC):
    """Abstract base class for online anomaly detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def learn_one(self, features: dict[str, float]) -> None:
        """Update the model with a single observation."""
        ...

    @abstractmethod
    def score_one(self, features: dict[str, float]) -> float:
        """Score a single observation (higher = more anomalous)."""
        ...

    def detect(self, features: dict[str, float], threshold: float = 0.5) -> DetectionResult:
        """Score and classify a single observation."""
        score = self.score_one(features)
        return DetectionResult(
            is_anomaly=score > threshold,
            anomaly_score=score,
            detector_name=self.name,
        )


# ─── 1. Half-Space Trees (River) ─────────────────────────────────────
class HalfSpaceTreesDetector(OnlineDetector):
    """Online anomaly detector using Half-Space Trees from River.

    Half-Space Trees work by randomly partitioning the feature space
    and counting observations in each partition. Anomalies fall in
    regions with low counts.

    This is one of the most effective online anomaly detection
    algorithms, requiring O(1) memory per tree node.

    Args:
        n_trees: Number of half-space trees.
        height: Maximum depth of each tree.
        window_size: Size of the reference window.
        seed: Random seed.
    """

    def __init__(
        self,
        n_trees: int = 25,
        height: int = 8,
        window_size: int = 250,
        seed: int = 42,
    ):
        from river.anomaly import HalfSpaceTrees

        self._model = HalfSpaceTrees(
            n_trees=n_trees,
            height=height,
            window_size=window_size,
            seed=seed,
        )
        self._samples_seen = 0

    @property
    def name(self) -> str:
        return "HalfSpaceTrees"

    def learn_one(self, features: dict[str, float]) -> None:
        self._model.learn_one(features)
        self._samples_seen += 1

    def score_one(self, features: dict[str, float]) -> float:
        # River's HST returns scores between 0 and 1
        # Higher score = more anomalous
        score = self._model.score_one(features)
        return float(score)


# ─── 2. EWMA Detector ────────────────────────────────────────────────
class EWMADetector(OnlineDetector):
    """Exponentially Weighted Moving Average anomaly detector.

    Tracks the EWMA of each feature and flags observations that
    deviate significantly from the running estimate. Adapts
    automatically to concept drift via exponential weighting.

    This is lightweight, interpretable, and works well for
    detecting sudden distribution shifts.

    Args:
        alpha: Smoothing factor (0 < alpha < 1). Lower = slower adaptation.
        threshold_sigmas: Number of standard deviations for anomaly threshold.
        warmup: Minimum samples before flagging anomalies.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        threshold_sigmas: float = 3.0,
        warmup: int = 100,
    ):
        self.alpha = alpha
        self.threshold_sigmas = threshold_sigmas
        self.warmup = warmup

        # Per-feature running statistics
        self._ewma: dict[str, float] = {}
        self._ewmvar: dict[str, float] = {}
        self._samples_seen = 0

    @property
    def name(self) -> str:
        return "EWMA"

    def learn_one(self, features: dict[str, float]) -> None:
        for key, value in features.items():
            if key not in self._ewma:
                self._ewma[key] = value
                self._ewmvar[key] = 0.0
            else:
                diff = value - self._ewma[key]
                self._ewma[key] += self.alpha * diff
                self._ewmvar[key] = (1 - self.alpha) * (
                    self._ewmvar[key] + self.alpha * diff * diff
                )
        self._samples_seen += 1

    def score_one(self, features: dict[str, float]) -> float:
        if self._samples_seen < self.warmup:
            return 0.0

        max_zscore = 0.0
        for key, value in features.items():
            if key in self._ewma:
                std = math.sqrt(max(self._ewmvar[key], 1e-10))
                zscore = abs(value - self._ewma[key]) / std
                max_zscore = max(max_zscore, zscore)

        # Normalize to [0, 1] using sigmoid-like transformation
        normalized = 1.0 / (1.0 + math.exp(-(max_zscore - self.threshold_sigmas)))
        return normalized


# ─── 3. Rolling Z-Score Detector ─────────────────────────────────────
class RollingZScoreDetector(OnlineDetector):
    """Rolling-window Z-score anomaly detector.

    Maintains a sliding window of recent values for each feature
    and flags observations that deviate significantly from the
    window's mean.

    Simple, interpretable, and effective for stationary-ish streams.

    Args:
        window_size: Number of recent values to track per feature.
        threshold: Z-score threshold for flagging anomalies.
    """

    def __init__(self, window_size: int = 200, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self._windows: dict[str, deque] = {}

    @property
    def name(self) -> str:
        return "RollingZScore"

    def learn_one(self, features: dict[str, float]) -> None:
        for key, value in features.items():
            if key not in self._windows:
                self._windows[key] = deque(maxlen=self.window_size)
            self._windows[key].append(value)

    def score_one(self, features: dict[str, float]) -> float:
        max_zscore = 0.0

        for key, value in features.items():
            window = self._windows.get(key)
            if window is None or len(window) < 30:
                continue

            values = np.array(window)
            mean = values.mean()
            std = values.std()

            if std < 1e-10:
                continue

            zscore = abs(value - mean) / std
            max_zscore = max(max_zscore, zscore)

        # Normalize
        normalized = min(1.0, max(0.0, (max_zscore - 1.0) / (self.threshold * 2)))
        return normalized


# ─── 4. Ensemble Detector ────────────────────────────────────────────
class EnsembleDetector(OnlineDetector):
    """Ensemble of multiple online anomaly detectors.

    Combines scores from multiple detectors using one of:
    - majority_vote: Anomaly if majority of detectors agree
    - weighted_average: Weighted average of anomaly scores
    - any: Anomaly if ANY detector flags it

    Args:
        detectors: List of OnlineDetector instances.
        method: Combination method.
        threshold: Score threshold for individual detectors.
        min_agree: Minimum detectors that must agree (for majority_vote).
    """

    def __init__(
        self,
        detectors: list[OnlineDetector],
        method: str = "majority_vote",
        threshold: float = 0.5,
        min_agree: int = 2,
    ):
        self.detectors = detectors
        self.method = method
        self.threshold = threshold
        self.min_agree = min_agree

    @property
    def name(self) -> str:
        return f"Ensemble({self.method})"

    def learn_one(self, features: dict[str, float]) -> None:
        for detector in self.detectors:
            detector.learn_one(features)

    def score_one(self, features: dict[str, float]) -> float:
        scores = [d.score_one(features) for d in self.detectors]

        if self.method == "weighted_average":
            return sum(scores) / len(scores)

        elif self.method == "any":
            return max(scores)

        else:  # majority_vote
            votes = sum(1 for s in scores if s > self.threshold)
            return votes / len(scores)

    def detect(self, features: dict[str, float], threshold: float = 0.5) -> DetectionResult:
        """Detect with per-detector details."""
        individual_results = []
        for detector in self.detectors:
            result = detector.detect(features, threshold)
            individual_results.append(result)

        ensemble_score = self.score_one(features)

        if self.method == "majority_vote":
            anomaly_votes = sum(1 for r in individual_results if r.is_anomaly)
            is_anomaly = anomaly_votes >= self.min_agree
        elif self.method == "any":
            is_anomaly = any(r.is_anomaly for r in individual_results)
        else:
            is_anomaly = ensemble_score > threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=ensemble_score,
            detector_name=self.name,
            details={
                "individual_scores": {
                    r.detector_name: round(r.anomaly_score, 4)
                    for r in individual_results
                },
                "individual_flags": {
                    r.detector_name: r.is_anomaly
                    for r in individual_results
                },
            },
        )


# ─── Factory ──────────────────────────────────────────────────────────
def build_detector_from_config(config: dict) -> EnsembleDetector:
    """Build an ensemble detector from a config dictionary.

    Args:
        config: The 'detectors' section of config.yaml.

    Returns:
        Configured EnsembleDetector.
    """
    detectors: list[OnlineDetector] = []

    hst_config = config.get("half_space_trees", {})
    if hst_config.get("enabled", True):
        detectors.append(HalfSpaceTreesDetector(
            n_trees=hst_config.get("n_trees", 25),
            height=hst_config.get("height", 8),
            window_size=hst_config.get("window_size", 250),
            seed=hst_config.get("seed", 42),
        ))

    ewma_config = config.get("ewma", {})
    if ewma_config.get("enabled", True):
        detectors.append(EWMADetector(
            alpha=ewma_config.get("alpha", 0.05),
            threshold_sigmas=ewma_config.get("threshold_sigmas", 3.0),
            warmup=ewma_config.get("warmup_period", 100),
        ))

    zscore_config = config.get("zscore", {})
    if zscore_config.get("enabled", True):
        detectors.append(RollingZScoreDetector(
            window_size=zscore_config.get("window_size", 200),
            threshold=zscore_config.get("threshold", 3.0),
        ))

    ensemble_config = config.get("ensemble", {})
    return EnsembleDetector(
        detectors=detectors,
        method=ensemble_config.get("method", "majority_vote"),
        min_agree=ensemble_config.get("min_detectors_agree", 2),
    )
