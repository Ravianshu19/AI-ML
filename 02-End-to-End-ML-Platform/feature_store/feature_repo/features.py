"""Feast feature view definitions for customer churn prediction.

Each FeatureView maps an Entity to a DataSource and defines
the features available for that entity. Features are versioned
and point-in-time joined during retrieval.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64, String

from data_sources import (
    customer_demographics_source,
    customer_behavior_source,
    customer_billing_source,
)

# ─── Entity Definition ───────────────────────────────────────────────
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="A unique customer in the system.",
)

# ─── Feature View: Demographics ──────────────────────────────────────
customer_demographics_fv = FeatureView(
    name="customer_demographics",
    entities=[customer],
    ttl=timedelta(days=365),  # Features valid for 1 year
    schema=[
        Field(name="age", dtype=Int64, description="Customer age in years"),
        Field(name="gender", dtype=String, description="Customer gender"),
        Field(name="tenure_months", dtype=Int64, description="Months as customer"),
        Field(name="contract_type", dtype=String, description="Month-to-month, One year, Two year"),
        Field(name="senior_citizen", dtype=Int64, description="1 if senior citizen, 0 otherwise"),
    ],
    source=customer_demographics_source,
    online=True,
    tags={"team": "data-science", "domain": "customer"},
)

# ─── Feature View: Behavior ─────────────────────────────────────────
customer_behavior_fv = FeatureView(
    name="customer_behavior",
    entities=[customer],
    ttl=timedelta(days=90),  # Behavior features refresh every 90 days
    schema=[
        Field(name="monthly_logins", dtype=Int64, description="Number of logins per month"),
        Field(name="monthly_transactions", dtype=Int64, description="Transactions per month"),
        Field(name="support_tickets_30d", dtype=Int64, description="Support tickets in last 30 days"),
        Field(name="avg_session_duration_min", dtype=Float32, description="Avg session duration in minutes"),
        Field(name="days_since_last_login", dtype=Int64, description="Days since last login"),
        Field(name="product_usage_score", dtype=Float32, description="Product engagement score 0-100"),
    ],
    source=customer_behavior_source,
    online=True,
    tags={"team": "data-science", "domain": "behavior"},
)

# ─── Feature View: Billing ───────────────────────────────────────────
customer_billing_fv = FeatureView(
    name="customer_billing",
    entities=[customer],
    ttl=timedelta(days=30),  # Billing refreshes monthly
    schema=[
        Field(name="monthly_charges", dtype=Float32, description="Monthly charge amount"),
        Field(name="total_charges", dtype=Float32, description="Total charges to date"),
        Field(name="payment_method", dtype=String, description="Payment method used"),
        Field(name="late_payments_6m", dtype=Int64, description="Late payments in last 6 months"),
        Field(name="has_autopay", dtype=Int64, description="1 if autopay enabled"),
    ],
    source=customer_billing_source,
    online=True,
    tags={"team": "data-science", "domain": "billing"},
)
