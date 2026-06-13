"""Real-time Streamlit dashboard for streaming anomaly detection.

Provides live visualization of:
- Time-series sensor/log data with anomaly highlights
- Per-detector anomaly scores
- Detection statistics (precision, recall, F1)
- Recent alert feed

Usage:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.stream.sensor_stream import SensorStream
from src.stream.log_stream import LogStream
from src.detectors.online_detectors import build_detector_from_config


# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔍 Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
)


def load_config() -> dict:
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# ─── Session State Initialization ─────────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.config = load_config()
    st.session_state.detector = build_detector_from_config(
        st.session_state.config.get("detectors", {})
    )
    st.session_state.data_buffer = deque(maxlen=500)
    st.session_state.score_buffer = deque(maxlen=500)
    st.session_state.alerts = deque(maxlen=50)
    st.session_state.stats = {
        "total": 0, "detected": 0, "true_anomalies": 0,
        "tp": 0, "fp": 0, "fn": 0,
    }
    st.session_state.running = False


# ─── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")

source_type = st.sidebar.selectbox(
    "Data Source",
    ["Sensor Stream", "Log Stream"],
)

n_samples = st.sidebar.slider(
    "Samples to process",
    min_value=100,
    max_value=10000,
    value=2000,
    step=100,
)

anomaly_prob = st.sidebar.slider(
    "Anomaly probability",
    min_value=0.0,
    max_value=0.15,
    value=0.03,
    step=0.005,
    format="%.3f",
)

threshold = st.sidebar.slider(
    "Detection threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
)

start_btn = st.sidebar.button("▶️ Start Detection", type="primary", use_container_width=True)
reset_btn = st.sidebar.button("🔄 Reset", use_container_width=True)

if reset_btn:
    st.session_state.data_buffer.clear()
    st.session_state.score_buffer.clear()
    st.session_state.alerts.clear()
    st.session_state.stats = {
        "total": 0, "detected": 0, "true_anomalies": 0,
        "tp": 0, "fp": 0, "fn": 0,
    }
    st.session_state.detector = build_detector_from_config(
        st.session_state.config.get("detectors", {})
    )
    st.rerun()


# ─── Header ──────────────────────────────────────────────────────────
st.title("🔍 Real-Time Anomaly Detection Dashboard")
st.markdown(
    "Online learning anomaly detection on streaming sensor/log data. "
    "Models update incrementally with each data point — **no batch retraining needed.**"
)

# ─── Main Content ─────────────────────────────────────────────────────
col_chart, col_stats = st.columns([3, 1])

# Metric cards
stats = st.session_state.stats
tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

with col_stats:
    st.metric("Total Processed", f"{stats['total']:,}")
    st.metric("Anomalies Detected", f"{stats['detected']:,}")
    st.metric("Precision", f"{precision:.3f}")
    st.metric("Recall", f"{recall:.3f}")
    st.metric("F1 Score", f"{f1:.3f}")

with col_chart:
    chart_placeholder = st.empty()

    if st.session_state.data_buffer:
        df = pd.DataFrame(list(st.session_state.data_buffer))

        fig = go.Figure()

        # Main signal
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df["value"] if "value" in df.columns else df.iloc[:, 0],
            mode="lines",
            name="Signal",
            line=dict(color="#4FC3F7", width=1),
        ))

        # Anomaly scores
        if st.session_state.score_buffer:
            scores = list(st.session_state.score_buffer)
            fig.add_trace(go.Scatter(
                x=list(range(len(scores))),
                y=[s * df["value"].max() if "value" in df.columns else s for s in scores],
                mode="lines",
                name="Anomaly Score",
                line=dict(color="#FF7043", width=1, dash="dot"),
                yaxis="y2",
            ))

        # Mark detected anomalies
        if "detected" in df.columns:
            anomaly_mask = df["detected"]
            if anomaly_mask.any():
                fig.add_trace(go.Scatter(
                    x=[i for i, v in enumerate(anomaly_mask) if v],
                    y=[df.iloc[i]["value"] if "value" in df.columns else 0
                       for i, v in enumerate(anomaly_mask) if v],
                    mode="markers",
                    name="Detected Anomaly",
                    marker=dict(color="red", size=8, symbol="x"),
                ))

        fig.update_layout(
            title="Live Signal + Anomaly Detection",
            xaxis_title="Sample",
            yaxis_title="Value",
            height=400,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis2=dict(title="Score", overlaying="y", side="right", range=[0, 1]),
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        chart_placeholder.info("Click **Start Detection** to begin processing data.")

# ─── Alerts Feed ──────────────────────────────────────────────────────
st.subheader("🚨 Recent Alerts")
if st.session_state.alerts:
    alert_df = pd.DataFrame(list(st.session_state.alerts))
    st.dataframe(alert_df, use_container_width=True, height=200)
else:
    st.info("No alerts yet.")


# ─── Processing Loop ─────────────────────────────────────────────────
if start_btn:
    detector = st.session_state.detector

    if source_type == "Sensor Stream":
        stream = SensorStream(anomaly_prob=anomaly_prob)
        progress = st.progress(0, text="Processing sensor stream...")

        for i, reading in enumerate(stream.generate(max_samples=n_samples)):
            features = reading.features()
            result = detector.detect(features, threshold)
            detector.learn_one(features)

            # Update stats
            st.session_state.stats["total"] += 1
            if reading.is_anomaly:
                st.session_state.stats["true_anomalies"] += 1
            if result.is_anomaly:
                st.session_state.stats["detected"] += 1
                if reading.is_anomaly:
                    st.session_state.stats["tp"] += 1
                else:
                    st.session_state.stats["fp"] += 1

                st.session_state.alerts.append({
                    "time": reading.timestamp.split("T")[1][:8],
                    "source": reading.sensor_id,
                    "score": round(result.anomaly_score, 3),
                    "type": reading.anomaly_type or "unknown",
                })
            elif reading.is_anomaly:
                st.session_state.stats["fn"] += 1

            st.session_state.data_buffer.append({
                **features,
                "detected": result.is_anomaly,
                "ground_truth": reading.is_anomaly,
            })
            st.session_state.score_buffer.append(result.anomaly_score)

            if (i + 1) % 50 == 0:
                progress.progress((i + 1) / n_samples, text=f"Processed {i+1:,}/{n_samples:,}")

        progress.progress(1.0, text="✅ Complete!")
        st.rerun()

    else:  # Log Stream
        stream = LogStream(anomaly_prob=anomaly_prob)
        progress = st.progress(0, text="Processing log stream...")

        for i, event in enumerate(stream.generate(max_samples=n_samples)):
            features = event.features()
            result = detector.detect(features, threshold)
            detector.learn_one(features)

            st.session_state.stats["total"] += 1
            if event.is_anomaly:
                st.session_state.stats["true_anomalies"] += 1
            if result.is_anomaly:
                st.session_state.stats["detected"] += 1
                if event.is_anomaly:
                    st.session_state.stats["tp"] += 1
                else:
                    st.session_state.stats["fp"] += 1

                st.session_state.alerts.append({
                    "time": event.timestamp.split("T")[1][:8],
                    "source": event.service,
                    "score": round(result.anomaly_score, 3),
                    "type": event.anomaly_type or "unknown",
                })
            elif event.is_anomaly:
                st.session_state.stats["fn"] += 1

            st.session_state.data_buffer.append({
                "value": features["response_time_ms"],
                **features,
                "detected": result.is_anomaly,
                "ground_truth": event.is_anomaly,
            })
            st.session_state.score_buffer.append(result.anomaly_score)

            if (i + 1) % 50 == 0:
                progress.progress((i + 1) / n_samples, text=f"Processed {i+1:,}/{n_samples:,}")

        progress.progress(1.0, text="✅ Complete!")
        st.rerun()
