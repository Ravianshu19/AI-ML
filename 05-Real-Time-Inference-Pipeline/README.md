# Real-Time Fraud Detection Inference Pipeline

A production-style real-time ML inference pipeline:

```
Producer (synthetic txns)
      │
      ▼
  Kafka / Redpanda  (topic: transactions)
      │
      ▼
  Consumer ──────► Redis (rolling per-user features: tx_count_1h/24h, avg_amount_24h)
      │
      ▼
  TorchServe  (custom handler, fraud-detection MLP)
      │
      ▼
  Postgres / TimescaleDB (table: scores) ──► Kafka (topic: scores)
      │
      ▼
  Streamlit Dashboard (throughput, latency p50/p95/p99, fraud rate, drift check)
```

## Components

| Component   | Tech                                  | Purpose                                              |
|-------------|---------------------------------------|-------------------------------------------------------|
| `model/`    | PyTorch                               | Trains a small MLP fraud classifier on synthetic data, exports TorchScript |
| `serving/`  | TorchServe + custom handler           | Serves the model over REST (`/predictions/fraud`)     |
| `producer/` | Python + kafka-python                 | Simulates a live stream of transaction events          |
| `consumer/` | Python + Redis + Kafka + Postgres     | Feature engineering, calls model, persists results     |
| `dashboard/`| Streamlit + Plotly                    | Live monitoring: throughput, latency, drift, fraud rate|

## Feature Vector (8 dims)

| # | Feature              | Source                                     |
|---|----------------------|---------------------------------------------|
| 0 | amount / 1000        | from event                                   |
| 1 | hour_of_day / 24     | from event timestamp                         |
| 2 | tx_count_1h / 10     | rolling count, Redis sorted set              |
| 3 | tx_count_24h / 50    | rolling count, Redis sorted set              |
| 4 | avg_amount_24h / 1000| rolling average, Redis sorted set            |
| 5 | merchant_risk_score  | from event                                   |
| 6 | is_foreign           | from event                                   |
| 7 | device_change        | from event                                   |

The same Redis sorted-set logic is used to compute `tx_count_1h`,
`tx_count_24h`, and `avg_amount_24h` on the fly in the consumer
(`compute_rolling_features`), giving each event a feature vector that
reflects the user's recent behavior.

## Quickstart

```bash
git clone <this-repo>
cd AI-ML/10-Real-Time-Inference-Pipeline

# (re)train the model — already included, but you can regenerate:
pip install torch --break-system-packages
python3 model/train.py

# build + run everything
docker compose up --build
```

Then open:
- **Dashboard**: http://localhost:8501
- **TorchServe inference API**: http://localhost:8080/predictions/fraud
- **TorchServe management API**: http://localhost:8081/models
- **TorchServe metrics**: http://localhost:8082/metrics

Give it ~30-60 seconds for all services to come up (TorchServe takes the
longest to build/start). The producer starts emitting ~5 events/sec
immediately; the dashboard auto-refreshes every 5 seconds.

## Manually testing the model endpoint

```bash
curl -X POST http://localhost:8080/predictions/fraud \
  -H "Content-Type: application/json" \
  -d '{"features": [0.8, 0.08, 0.3, 0.4, 0.07, 0.6, 1, 1]}'
```

Expected response:
```json
{"fraud_probability": 0.91, "is_fraud": true}
```

## Latency Benchmarking

Each scored transaction records three latency measurements (in `scores`
table / dashboard):

- `latency_feature_ms` — Redis round-trip for rolling feature computation
- `latency_inference_ms` — HTTP round-trip to TorchServe
- `latency_total_ms` — end-to-end (Kafka consume → Postgres-ready row)

On a local Docker Compose setup (single CPU container, batch size 1),
typical numbers are:

| Stage              | p50    | p95    | p99    |
|--------------------|--------|--------|--------|
| Feature engineering| ~1-3ms | ~5ms   | ~10ms  |
| Inference (TorchServe REST)| ~5-15ms| ~25ms | ~40ms |
| End-to-end         | ~10-25ms| ~35ms | ~55ms |

(Exact numbers depend on host hardware — the dashboard computes these
live from the `scores` table.)

## Drift Monitoring

The dashboard compares the live rolling average of `amount` against the
training-set reference mean (computed from the synthetic data generator's
log-normal parameters). If the live average deviates by more than 25%, a
drift warning is shown. This is a simple stand-in for more rigorous drift
tests (PSI, KS-test, etc.) that would be used in a production system —
swapping in `scipy.stats.ks_2samp` against a stored reference sample is a
natural next step.

## Scaling Notes / Next Steps

- **Throughput**: increase `EVENTS_PER_SEC` in `docker-compose.yml`
  (producer service) to stress-test.
- **TorchServe**: `serving/config.properties` sets `maxWorkers: 2` and
  `batchSize: 1`. Increasing batch size + max batch delay would improve
  throughput at the cost of per-request latency — a classic latency/
  throughput tradeoff worth measuring.
- **Triton alternative**: the model is exported as TorchScript, so it
  could equally be served via NVIDIA Triton Inference Server with an ONNX
  or TorchScript backend — useful if GPU serving / dynamic batching across
  models is needed.
- **Schema registry**: for a more "production" feel, add Avro/Protobuf
  schemas via the Redpanda/Confluent schema registry instead of raw JSON.
- **Alerting**: hook the `scores` Kafka topic into a simple alerting
  consumer (e.g., Slack webhook when fraud rate spikes or latency p99
  exceeds a threshold).

## Repo Structure

```
10-Real-Time-Inference-Pipeline/
├── model/
│   ├── train.py          # trains + exports fraud_model.pt
│   └── fraud_model.pt     # TorchScript model (generated)
├── serving/
│   ├── handler.py         # TorchServe custom handler
│   ├── Dockerfile
│   └── config.properties
├── producer/
│   ├── producer.py
│   ├── Dockerfile
│   └── requirements.txt
├── consumer/
│   ├── consumer.py
│   ├── Dockerfile
│   └── requirements.txt
├── dashboard/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── README.md
```
