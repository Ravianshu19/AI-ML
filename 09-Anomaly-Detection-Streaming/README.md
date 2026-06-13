# 🔍 Anomaly Detection on Streaming Data — Online Learning

Real-time anomaly detection on streaming sensor and log data using **online (incremental) learning** algorithms that learn from each data point without storing the full dataset.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Online Learning** | Models update incrementally with each data point — O(1) memory |
| **Half-Space Trees** | River's streaming anomaly detector using random space partitioning |
| **EWMA** | Exponentially Weighted Moving Average for drift-aware detection |
| **Ensemble** | Majority-vote across multiple detectors for robust detection |
| **Concept Drift** | Gradual distribution shifts that the models adapt to automatically |

## Architecture

```
┌─────────────────────┐     ┌─────────────────────────┐
│  Data Streams        │     │  Online Detectors        │
│  ├─ Sensor Stream    │────►│  ├─ Half-Space Trees     │
│  │  (IoT readings)   │     │  ├─ EWMA Detector        │
│  └─ Log Stream       │     │  └─ Rolling Z-Score      │
│    (microservices)   │     │         │                 │
└─────────────────────┘     │    Ensemble (vote)        │
                             └──────────┬──────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
             ┌──────────┐      ┌──────────────┐    ┌──────────┐
             │  Alerts   │      │  Statistics  │    │Dashboard │
             │  Manager  │      │  Tracker     │    │(Streamlit│
             └──────────┘      └──────────────┘    └──────────┘
```

## Quick Start

```bash
cd 09-Anomaly-Detection-Streaming
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run on sensor data (live terminal dashboard)
python -m src.main sensor --samples 5000

# Run on log data
python -m src.main logs --samples 3000 --anomaly-prob 0.05

# Benchmark all detectors
python -m src.main benchmark --samples 10000

# Launch Streamlit dashboard
python -m src.main dashboard
```

## Project Structure

```
09-Anomaly-Detection-Streaming/
├── config.yaml                        # Pipeline configuration
├── requirements.txt
├── src/
│   ├── main.py                        # 🎯 CLI entry point
│   ├── stream/
│   │   ├── sensor_stream.py           # IoT sensor simulator
│   │   └── log_stream.py             # Microservice log simulator
│   ├── detectors/
│   │   └── online_detectors.py        # 3 detectors + ensemble
│   ├── pipeline/
│   │   └── streaming_pipeline.py      # Main processing pipeline
│   └── dashboard/
│       └── app.py                     # Streamlit real-time UI
```

## Detectors

### 1. Half-Space Trees (River)
- Randomly partitions feature space into half-spaces
- Counts observations per partition; anomalies fall in sparse regions
- O(1) memory per tree, no retraining needed

### 2. EWMA (Exponentially Weighted Moving Average)
- Tracks running mean/variance per feature with exponential decay
- Flags deviations beyond N standard deviations
- Naturally adapts to concept drift via exponential weighting

### 3. Rolling Z-Score
- Sliding window of recent values per feature
- Flags observations with Z-score above threshold
- Simple, interpretable baseline

### 4. Ensemble (Majority Vote)
- Combines all three detectors
- Anomaly only if ≥2 detectors agree (reduces false positives)

## Anomaly Types Simulated

### Sensor Stream
| Type | Description |
|------|-------------|
| `point_spike` | Sudden upward spike in a single feature |
| `point_drop` | Sudden downward drop |
| `multi_feature` | Multiple features shift simultaneously |
| `frozen` | Sensor stuck at a constant value |

### Log Stream
| Type | Description |
|------|-------------|
| `latency_spike` | Response time increases 5-20x |
| `error_burst` | Error rate jumps to 15-60% |
| `traffic_surge` | Request count increases 5-15x |
| `memory_leak` | Memory usage doubles+ |
| `cpu_spike` | CPU usage hits 85-99% |
| `cascade_failure` | Multiple metrics degrade simultaneously |

## License

MIT
