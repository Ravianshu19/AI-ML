"""Integration tests for the ML pipeline components.

These tests validate that the pipeline components work correctly
without requiring external services (MLflow, etc.).

Usage:
    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ─── Feature Generation Tests ───────────────────────────────────────
class TestFeatureGeneration:
    """Test synthetic data generation."""

    def test_demographics_shape(self):
        from feature_store.populate_features import generate_customer_demographics
        df = generate_customer_demographics(100)
        assert len(df) == 100
        assert "customer_id" in df.columns
        assert "age" in df.columns
        assert "tenure_months" in df.columns

    def test_demographics_ranges(self):
        from feature_store.populate_features import generate_customer_demographics
        df = generate_customer_demographics(1000)
        assert df["age"].min() >= 18
        assert df["age"].max() <= 79
        assert df["senior_citizen"].isin([0, 1]).all()

    def test_behavior_shape(self):
        from feature_store.populate_features import generate_customer_behavior
        df = generate_customer_behavior(100)
        assert len(df) == 100
        assert "monthly_logins" in df.columns
        assert "product_usage_score" in df.columns

    def test_billing_shape(self):
        from feature_store.populate_features import generate_customer_billing
        df = generate_customer_billing(100)
        assert len(df) == 100
        assert "monthly_charges" in df.columns
        assert df["monthly_charges"].min() >= 0

    def test_training_dataset_labels(self):
        from feature_store.populate_features import generate_training_dataset
        df = generate_training_dataset(500)
        assert "churn" in df.columns
        assert set(df["churn"].unique()).issubset({0, 1})
        # Churn rate should be reasonable (10-50%)
        churn_rate = df["churn"].mean()
        assert 0.05 < churn_rate < 0.60


# ─── Model Training Tests ───────────────────────────────────────────
class TestModelTraining:
    """Test model training functionality."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            "age": np.random.randint(18, 80, n),
            "tenure_months": np.random.randint(1, 72, n),
            "monthly_charges": np.random.uniform(20, 120, n),
            "support_tickets_30d": np.random.poisson(1, n),
        })
        y = pd.Series((np.random.random(n) > 0.7).astype(int), name="churn")
        return X, y

    def test_model_trains(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_model_predicts(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_model_accuracy_above_baseline(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        accuracy = accuracy_score(y, model.predict(X))
        # Should beat random baseline
        assert accuracy > 0.5


# ─── Drift Detection Tests ───────────────────────────────────────────
class TestDriftDetection:
    """Test drift monitoring functionality."""

    def test_no_drift_identical_data(self):
        from monitoring.drift_detector import simulate_production_data
        reference = pd.DataFrame({
            "monthly_charges": np.random.uniform(20, 120, 500),
            "support_tickets_30d": np.random.poisson(1, 500),
            "days_since_last_login": np.random.randint(0, 60, 500),
        })
        production = simulate_production_data(reference, drift_level=0.0)
        # With zero drift, data should be very similar
        for col in reference.columns:
            diff = abs(reference[col].mean() - production[col].mean())
            assert diff < reference[col].std() * 2

    def test_high_drift_changes_data(self):
        from monitoring.drift_detector import simulate_production_data
        reference = pd.DataFrame({
            "monthly_charges": np.random.uniform(20, 120, 500),
            "support_tickets_30d": np.random.poisson(1, 500),
            "days_since_last_login": np.random.randint(0, 60, 500),
        })
        production = simulate_production_data(reference, drift_level=0.8)
        # With high drift, means should differ
        diff = abs(
            reference["monthly_charges"].mean() -
            production["monthly_charges"].mean()
        )
        assert diff > 1.0  # Should show meaningful difference


# ─── API Schema Tests ────────────────────────────────────────────────
class TestAPISchemas:
    """Test API request/response schemas."""

    def test_customer_features_schema(self):
        from serving.app import CustomerFeatures
        features = CustomerFeatures(
            age=35, tenure_months=24, senior_citizen=0,
            monthly_logins=15, monthly_transactions=8,
            support_tickets_30d=1, avg_session_duration_min=12.5,
            days_since_last_login=3, product_usage_score=72.5,
            monthly_charges=65.0, total_charges=1560.0,
            late_payments_6m=0, has_autopay=1,
        )
        assert features.age == 35
        assert features.monthly_charges == 65.0

    def test_invalid_age_rejected(self):
        from serving.app import CustomerFeatures
        with pytest.raises(Exception):
            CustomerFeatures(
                age=10,  # Below 18 — should fail
                tenure_months=24, senior_citizen=0,
                monthly_logins=15, monthly_transactions=8,
                support_tickets_30d=1, avg_session_duration_min=12.5,
                days_since_last_login=3, product_usage_score=72.5,
                monthly_charges=65.0, total_charges=1560.0,
                late_payments_6m=0, has_autopay=1,
            )
