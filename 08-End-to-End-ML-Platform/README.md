# 🏭 End-to-End ML Platform

A production-grade ML platform for **customer churn prediction** featuring all four pillars of MLOps:

| Component | Tool | Purpose |
|-----------|------|---------|
| **Feature Store** | Feast | Centralized feature management, point-in-time joins, online/offline serving |
| **Model Registry** | MLflow | Experiment tracking, model versioning, stage promotion |
| **CI/CD** | GitHub Actions | Automated train → validate → deploy pipeline |
| **Drift Monitoring** | Evidently AI | Data drift detection, data quality checks, automated alerts |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    📊 DATA LAYER                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐   │
│  │ Raw Data     │───►│ Feature Store    │───►│ Training     │   │
│  │ (CSV/Parquet)│    │ (Feast)          │    │ Data         │   │
│  └──────────────┘    └──────────────────┘    └──────┬───────┘   │
│                                                      │          │
├──────────────────────────────────────────────────────┼──────────┤
│                    🏋️ TRAINING LAYER                  │          │
│  ┌──────────────────────────────────────────────┐    │          │
│  │  MLflow Experiment Tracking                   │◄──┘          │
│  │  • Log params, metrics, artifacts             │              │
│  │  • Model versioning & registry               │              │
│  │  • Stage promotion (Staging → Production)     │              │
│  └──────────────────────┬───────────────────────┘              │
│                          │                                      │
├──────────────────────────┼──────────────────────────────────────┤
│                    🚀 SERVING LAYER               │              │
│  ┌──────────────────────▼───────────────────────┐              │
│  │  FastAPI Model Server                         │              │
│  │  • /predict (single)  • /predict/batch        │              │
│  │  • /health            • /reload               │              │
│  │  • Auto-loads from MLflow Production stage    │              │
│  └──────────────────────┬───────────────────────┘              │
│                          │                                      │
├──────────────────────────┼──────────────────────────────────────┤
│                    📡 MONITORING LAYER            │              │
│  ┌──────────────────────▼───────────────────────┐              │
│  │  Evidently AI Drift Monitoring                │              │
│  │  • Data drift detection (per-feature)         │              │
│  │  • Data quality reports                       │              │
│  │  • Automated test suites (pass/fail)          │              │
│  │  • HTML dashboard reports                     │              │
│  └──────────────────────────────────────────────┘              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    🔄 CI/CD (GitHub Actions)                     │
│  validate-data → train → evaluate → deploy → monitor            │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install

```bash
cd 08-End-to-End-ML-Platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
# Run all stages: features → train → evaluate → monitor
python -m pipeline --stage all

# Or run individual stages
python -m pipeline --stage features      # Generate synthetic data
python -m pipeline --stage train -m gradient_boosting --register
python -m pipeline --stage evaluate      # Compare MLflow runs
python -m pipeline --stage monitor -d 0.3  # Test with 30% drift
python -m pipeline --stage serve         # Start FastAPI server
```

### 3. With MLflow Server (optional)

```bash
# Start MLflow in a separate terminal
mlflow server --host 0.0.0.0 --port 5000

# Or use Docker Compose
docker compose up -d mlflow
```

### 4. With Docker

```bash
docker compose up -d  # Starts MLflow + Model Server
```

## Project Structure

```
08-End-to-End-ML-Platform/
├── config.yaml                    # Central pipeline configuration
├── requirements.txt               # Python dependencies
├── docker-compose.yml             # Docker services (MLflow + Server)
├── pipeline.py                    # 🎯 Main orchestrator CLI
│
├── feature_store/                 # 📦 Feast Feature Store
│   ├── feature_repo/
│   │   ├── feature_store.yaml     # Feast project config
│   │   ├── data_sources.py        # Data source definitions
│   │   └── features.py            # Feature view definitions
│   └── populate_features.py       # Synthetic data generator
│
├── training/                      # 🏋️ Model Training + MLflow
│   ├── train.py                   # Training pipeline with MLflow tracking
│   └── evaluate.py                # Model comparison & promotion
│
├── serving/                       # 🚀 FastAPI Model Server
│   ├── app.py                     # FastAPI application
│   └── Dockerfile                 # Container image
│
├── monitoring/                    # 📡 Evidently Drift Monitoring
│   └── drift_detector.py          # Drift detection + reporting
│
├── cicd/                          # 🔄 CI/CD
│   └── train_and_deploy.yml       # GitHub Actions workflow
│
└── tests/                         # 🧪 Tests
    └── test_pipeline.py           # Integration tests
```

## Components Deep Dive

### 📦 Feature Store (Feast)
- **3 Feature Views**: demographics, behavior, billing
- **Entity**: customer (join key: `customer_id`)
- **Online + Offline** serving support
- Point-in-time correct feature retrieval

### 🏋️ Model Registry (MLflow)
- **Experiment tracking**: params, metrics, artifacts
- **Model versioning**: automatic versioning on registration
- **Stage management**: Staging → Production promotion
- **3 model types**: Random Forest, Gradient Boosting, Logistic Regression

### 🚀 Model Serving (FastAPI)
- **Endpoints**: `/predict`, `/predict/batch`, `/health`, `/reload`
- **Auto-loads** Production model from MLflow registry
- **Pydantic validation** on all inputs
- **Risk scoring**: Low / Medium / High based on probability

### 📡 Drift Monitoring (Evidently AI)
- **Data drift**: Per-feature statistical tests
- **Data quality**: Missing values, duplicates, type checks
- **Test suites**: Automated pass/fail with configurable thresholds
- **HTML reports**: Saved to `monitoring/reports/`
- **Simulated drift**: Configurable drift levels for testing

### 🔄 CI/CD (GitHub Actions)
5-stage pipeline:
1. **Validate Data** — Generate features, run quality checks
2. **Train Model** — Train with MLflow tracking
3. **Evaluate & Promote** — Compare runs, promote best
4. **Deploy** — Build Docker image, push to registry
5. **Monitor** — Post-deploy drift detection

## License

MIT
