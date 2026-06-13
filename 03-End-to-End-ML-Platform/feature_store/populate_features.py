"""Generate synthetic feature data and materialize into the Feast feature store.

This script:
1. Generates realistic synthetic customer data (demographics, behavior, billing)
2. Saves as Parquet files for Feast offline store
3. Applies Feast feature definitions
4. Materializes features to the online store

Usage:
    python -m feature_store.populate_features
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

console = Console()
logger = logging.getLogger(__name__)

NUM_CUSTOMERS = 5000
DATA_DIR = Path("feature_store/feature_repo/data")


def generate_customer_demographics(n: int) -> pd.DataFrame:
    """Generate synthetic customer demographic data."""
    np.random.seed(42)
    now = datetime.now()

    data = {
        "customer_id": [f"CUST_{i:05d}" for i in range(n)],
        "age": np.random.randint(18, 80, size=n),
        "gender": np.random.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.48, 0.04]),
        "tenure_months": np.random.randint(1, 72, size=n),
        "contract_type": np.random.choice(
            ["Month-to-month", "One year", "Two year"],
            size=n,
            p=[0.5, 0.3, 0.2],
        ),
        "senior_citizen": (np.random.random(n) > 0.8).astype(int),
        "event_timestamp": [now - timedelta(hours=np.random.randint(0, 720)) for _ in range(n)],
        "created_at": [now - timedelta(days=np.random.randint(30, 365)) for _ in range(n)],
    }
    return pd.DataFrame(data)


def generate_customer_behavior(n: int) -> pd.DataFrame:
    """Generate synthetic customer behavior data."""
    np.random.seed(43)
    now = datetime.now()

    data = {
        "customer_id": [f"CUST_{i:05d}" for i in range(n)],
        "monthly_logins": np.random.poisson(15, size=n),
        "monthly_transactions": np.random.poisson(8, size=n),
        "support_tickets_30d": np.random.poisson(1, size=n),
        "avg_session_duration_min": np.round(np.random.exponential(12, size=n), 2).astype(np.float32),
        "days_since_last_login": np.random.randint(0, 60, size=n),
        "product_usage_score": np.round(np.random.beta(5, 2, size=n) * 100, 2).astype(np.float32),
        "event_timestamp": [now - timedelta(hours=np.random.randint(0, 720)) for _ in range(n)],
        "created_at": [now - timedelta(days=np.random.randint(1, 90)) for _ in range(n)],
    }
    return pd.DataFrame(data)


def generate_customer_billing(n: int) -> pd.DataFrame:
    """Generate synthetic customer billing data."""
    np.random.seed(44)
    now = datetime.now()

    monthly = np.round(np.random.uniform(20, 120, size=n), 2).astype(np.float32)
    tenure = np.random.randint(1, 72, size=n)

    data = {
        "customer_id": [f"CUST_{i:05d}" for i in range(n)],
        "monthly_charges": monthly,
        "total_charges": np.round(monthly * tenure, 2).astype(np.float32),
        "payment_method": np.random.choice(
            ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"],
            size=n,
            p=[0.35, 0.25, 0.25, 0.15],
        ),
        "late_payments_6m": np.random.poisson(0.5, size=n),
        "has_autopay": (np.random.random(n) > 0.4).astype(int),
        "event_timestamp": [now - timedelta(hours=np.random.randint(0, 720)) for _ in range(n)],
        "created_at": [now - timedelta(days=np.random.randint(1, 30)) for _ in range(n)],
    }
    return pd.DataFrame(data)


def generate_training_dataset(n: int) -> pd.DataFrame:
    """Generate a combined training dataset with churn labels."""
    np.random.seed(45)

    demographics = generate_customer_demographics(n)
    behavior = generate_customer_behavior(n)
    billing = generate_customer_billing(n)

    # Merge all features
    df = demographics[["customer_id", "age", "tenure_months", "senior_citizen"]].copy()
    df = df.merge(
        behavior[["customer_id", "monthly_logins", "monthly_transactions",
                  "support_tickets_30d", "avg_session_duration_min",
                  "days_since_last_login", "product_usage_score"]],
        on="customer_id",
    )
    df = df.merge(
        billing[["customer_id", "monthly_charges", "total_charges",
                 "late_payments_6m", "has_autopay"]],
        on="customer_id",
    )

    # Generate churn labels (correlated with features)
    churn_probability = (
        0.1
        + 0.15 * (df["tenure_months"] < 12).astype(float)
        + 0.10 * (df["support_tickets_30d"] > 2).astype(float)
        + 0.10 * (df["days_since_last_login"] > 30).astype(float)
        + 0.08 * (df["late_payments_6m"] > 1).astype(float)
        - 0.10 * (df["has_autopay"] == 1).astype(float)
        - 0.05 * (df["product_usage_score"] > 70).astype(float)
    )
    churn_probability = churn_probability.clip(0.05, 0.95)
    df["churn"] = (np.random.random(n) < churn_probability).astype(int)

    return df


def main() -> None:
    """Generate all data and save to disk."""
    console.print(Panel("[bold cyan]📦 Generating Feature Store Data[/bold cyan]", border_style="cyan"))

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Generate and save feature data
    console.print("  ⏳ Generating customer demographics...")
    demographics = generate_customer_demographics(NUM_CUSTOMERS)
    demographics.to_parquet(DATA_DIR / "customer_demographics.parquet", index=False)
    console.print(f"  ✅ Demographics: {len(demographics)} records")

    console.print("  ⏳ Generating customer behavior...")
    behavior = generate_customer_behavior(NUM_CUSTOMERS)
    behavior.to_parquet(DATA_DIR / "customer_behavior.parquet", index=False)
    console.print(f"  ✅ Behavior: {len(behavior)} records")

    console.print("  ⏳ Generating customer billing...")
    billing = generate_customer_billing(NUM_CUSTOMERS)
    billing.to_parquet(DATA_DIR / "customer_billing.parquet", index=False)
    console.print(f"  ✅ Billing: {len(billing)} records")

    # Generate training dataset
    console.print("  ⏳ Generating training dataset with labels...")
    training_df = generate_training_dataset(NUM_CUSTOMERS)
    training_df.to_csv("data/processed/training_data.csv", index=False)
    console.print(f"  ✅ Training data: {len(training_df)} records, churn rate: {training_df['churn'].mean():.1%}")

    # Save reference data for monitoring
    reference = training_df.sample(frac=0.3, random_state=42)
    reference.to_csv("data/processed/reference.csv", index=False)
    console.print(f"  ✅ Reference data: {len(reference)} records (for drift monitoring)")

    console.print()
    console.print("[bold green]✅ Feature data generation complete![/bold green]")


if __name__ == "__main__":
    main()
