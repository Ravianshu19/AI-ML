"""Feast data source definitions for customer churn prediction.

Defines the offline data sources that Feast uses to serve
historical features for training and batch scoring.
"""

from datetime import timedelta

from feast import FileSource

# ─── Customer Demographics Source ────────────────────────────────────
customer_demographics_source = FileSource(
    name="customer_demographics_source",
    path="data/customer_demographics.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Customer demographic information including age, gender, tenure.",
)

# ─── Customer Behavior Source ────────────────────────────────────────
customer_behavior_source = FileSource(
    name="customer_behavior_source",
    path="data/customer_behavior.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Customer usage behavior: logins, transactions, support tickets.",
)

# ─── Customer Billing Source ─────────────────────────────────────────
customer_billing_source = FileSource(
    name="customer_billing_source",
    path="data/customer_billing.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Customer billing data: monthly charges, total charges, payment method.",
)
