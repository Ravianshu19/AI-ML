"""
Real-time monitoring dashboard for the inference pipeline.

Shows:
  - Live throughput (events/min)
  - Latency percentiles (p50/p95/p99) for end-to-end inference latency
  - Fraud prediction distribution
  - Fraud rate over time
  - Simple feature drift indicator (live avg transaction amount vs. training
    reference mean)
"""

import os
import time

import pandas as pd
import psycopg2
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

PG_DSN = os.environ.get(
    "POSTGRES_DSN",
    "dbname=pipeline user=pipeline password=pipeline host=localhost port=5432",
)

# Reference stats from training data (model/train.py generate_synthetic_data)
# amount ~ Lognormal(mean=3.5, sigma=1.2) -> E[amount] = exp(mu + sigma^2/2)
TRAINING_REF_AVG_AMOUNT = 2.718281828 ** (3.5 + (1.2**2) / 2)  # ~ 68.0
DRIFT_THRESHOLD_PCT = 25.0  # flag if live avg differs by more than this %


st.set_page_config(page_title="Inference Pipeline Monitor", layout="wide")
st_autorefresh(interval=5000, key="refresh")

st.title("🔎 Real-Time Inference Pipeline Monitor")
st.caption("Kafka → Redis (feature store) → TorchServe → Postgres → Dashboard")


@st.cache_resource
def get_conn():
    for _ in range(20):
        try:
            return psycopg2.connect(PG_DSN)
        except Exception:
            time.sleep(2)
    raise RuntimeError("Could not connect to Postgres")


def load_recent(minutes=10):
    conn = get_conn()
    conn.rollback()
    query = f"""
        SELECT *
        FROM scores
        WHERE created_at > now() - interval '{minutes} minutes'
        ORDER BY created_at DESC
        LIMIT 5000
    """
    try:
        return pd.read_sql(query, conn)
    except Exception:
        conn.rollback()
        return pd.DataFrame()


df = load_recent()

if df.empty:
    st.info(
        "No data yet. Make sure the producer, consumer, TorchServe, and "
        "Postgres services are running. New scores will appear here "
        "automatically (refreshes every 5s)."
    )
    st.stop()

df["created_at"] = pd.to_datetime(df["created_at"])

# ---------------------------------------------------------------------------
# Top-level metrics
# ---------------------------------------------------------------------------
total_events = len(df)
window_seconds = max(
    (df["created_at"].max() - df["created_at"].min()).total_seconds(), 1
)
throughput = total_events / window_seconds * 60  # events/min

fraud_rate = df["is_fraud"].mean() * 100
avg_amount_live = df["amount"].mean()
drift_pct = (
    abs(avg_amount_live - TRAINING_REF_AVG_AMOUNT) / TRAINING_REF_AVG_AMOUNT * 100
)

p50 = df["latency_total_ms"].quantile(0.50)
p95 = df["latency_total_ms"].quantile(0.95)
p99 = df["latency_total_ms"].quantile(0.99)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Throughput (events/min)", f"{throughput:.1f}")
col2.metric("Fraud rate", f"{fraud_rate:.2f}%")
col3.metric("p95 latency (ms)", f"{p95:.1f}")
col4.metric("p99 latency (ms)", f"{p99:.1f}")

# ---------------------------------------------------------------------------
# Drift indicator
# ---------------------------------------------------------------------------
st.subheader("Feature Drift Check — Transaction Amount")
drift_col1, drift_col2 = st.columns([1, 3])
with drift_col1:
    st.metric(
        "Live avg amount",
        f"{avg_amount_live:.2f}",
        delta=f"{drift_pct:.1f}% vs training ref ({TRAINING_REF_AVG_AMOUNT:.1f})",
        delta_color="inverse" if drift_pct > DRIFT_THRESHOLD_PCT else "off",
    )
    if drift_pct > DRIFT_THRESHOLD_PCT:
        st.warning("⚠️ Possible distribution drift detected on `amount` feature.")
    else:
        st.success("Live amount distribution within normal range.")

with drift_col2:
    fig_amt = px.histogram(
        df, x="amount", nbins=40, title="Live distribution of transaction amounts"
    )
    fig_amt.add_vline(
        x=TRAINING_REF_AVG_AMOUNT,
        line_dash="dash",
        line_color="red",
        annotation_text="training ref mean",
    )
    st.plotly_chart(fig_amt, use_container_width=True)

# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
st.subheader("Latency Breakdown (last 10 min)")
lat_col1, lat_col2 = st.columns(2)

with lat_col1:
    fig_lat = px.histogram(
        df,
        x="latency_total_ms",
        nbins=40,
        title="End-to-end latency distribution (ms)",
    )
    st.plotly_chart(fig_lat, use_container_width=True)

with lat_col2:
    df_sorted = df.sort_values("created_at")
    fig_lat_time = px.line(
        df_sorted,
        x="created_at",
        y="latency_total_ms",
        title="Latency over time (ms)",
    )
    st.plotly_chart(fig_lat_time, use_container_width=True)

lat_stats = pd.DataFrame(
    {
        "stage": ["feature_engineering", "inference", "end_to_end"],
        "p50_ms": [
            df["latency_feature_ms"].quantile(0.5),
            df["latency_inference_ms"].quantile(0.5),
            p50,
        ],
        "p95_ms": [
            df["latency_feature_ms"].quantile(0.95),
            df["latency_inference_ms"].quantile(0.95),
            p95,
        ],
        "p99_ms": [
            df["latency_feature_ms"].quantile(0.99),
            df["latency_inference_ms"].quantile(0.99),
            p99,
        ],
    }
)
st.dataframe(lat_stats, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
st.subheader("Fraud Predictions")
pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    fig_prob = px.histogram(
        df, x="fraud_probability", nbins=40, title="Predicted fraud probability"
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with pred_col2:
    df_sorted["minute"] = df_sorted["created_at"].dt.floor("min")
    fraud_over_time = (
        df_sorted.groupby("minute")["is_fraud"].mean().reset_index()
    )
    fraud_over_time["is_fraud"] *= 100
    fig_fraud_time = px.line(
        fraud_over_time,
        x="minute",
        y="is_fraud",
        title="Fraud rate over time (%)",
        markers=True,
    )
    st.plotly_chart(fig_fraud_time, use_container_width=True)

# ---------------------------------------------------------------------------
# Recent flagged transactions
# ---------------------------------------------------------------------------
st.subheader("Most recent flagged transactions")
flagged = df[df["is_fraud"]].sort_values("created_at", ascending=False).head(20)
st.dataframe(
    flagged[
        [
            "created_at",
            "tx_id",
            "user_id",
            "amount",
            "fraud_probability",
            "tx_count_1h",
            "tx_count_24h",
            "latency_total_ms",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Showing {total_events} events from the last 10 minutes. "
    "Auto-refreshes every 5 seconds."
)
