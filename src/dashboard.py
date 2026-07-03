"""
dashboard.py - Streamlit Monitoring Dashboard & Data Drift Detector
Includes API operational metrics, score distributions, and Evidently AI-backed Data Drift reports.
"""

import os
import sqlite3
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from evidently import Report
from evidently.presets import DataDriftPreset

# Page config
st.set_page_config(
    page_title="Antibody Scoring - Production Monitoring Dashboard",
    page_icon="🧬",
    layout="wide"
)

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, "production_logs.db")
dataset_path = os.path.join(base_dir, "..", "data", "final_dataset.parquet")

# -----------------------------------------------------------------------------
# Database Utility & Data Seeding
# -----------------------------------------------------------------------------
def get_db_connection():
    return sqlite3.connect(db_path)

def seed_database_if_empty():
    """Seeds the production logs database with realistic calls to ensure the dashboard immediately works."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if logs table is empty
        cursor.execute("SELECT COUNT(*) FROM api_logs")
        count = cursor.fetchone()[0]
        
        if count == 0:
            st.info("💡 Production database is empty. Seeding with 50 realistic historical and drifted requests...")
            
            # Load some sequences from the final dataset
            if os.path.exists(dataset_path):
                ref_df = pd.read_parquet(dataset_path)
                
                # Seed some exact match calls (30 rows with standard values)
                exact_rows = ref_df.sample(min(30, len(ref_df)), random_state=42)
                now = datetime.datetime.now()
                
                for i, (_, row) in enumerate(exact_rows.iterrows()):
                    # Simulate standard request
                    latency = float(np.random.normal(loc=12.0, scale=3.0)) # 12ms average
                    latency = max(1.0, latency)
                    time_offset = datetime.timedelta(hours=(48 - i))
                    timestamp = (now - time_offset).strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Target category to simulated probability
                    prob = 0.85 if row["hic_category"] == "high_hic" else 0.12
                    prob = float(np.clip(prob + np.random.normal(0, 0.05), 0, 1))
                    pred = 1 if prob >= 0.5 else 0
                    
                    cursor.execute("""
                        INSERT INTO api_logs (timestamp, antibody_id, vh, vl, tm_app, psr, prediction, probability, latency_ms, status_code)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 200)
                    """, (
                        timestamp, row["antibody_id"], row["vh"], row["vl"], 
                        float(row["tm_app"]), float(row["psr"]), pred, prob, latency
                    ))
                
                # Seed some drift calls (20 rows with drifted values - e.g. lower tm_app and higher psr)
                drifted_rows = ref_df.sample(min(20, len(ref_df)), random_state=24)
                for i, (_, row) in enumerate(drifted_rows.iterrows()):
                    # Simulate drift: lower apparent melting temperature and much higher psr score
                    drifted_tm = float(row["tm_app"] - 8.0) # -8°C drift
                    drifted_psr = float(row["psr"] + 2.5) # +2.5 PSR drift
                    
                    latency = float(np.random.normal(loc=15.0, scale=5.0))
                    latency = max(1.0, latency)
                    time_offset = datetime.timedelta(hours=(20 - i))
                    timestamp = (now - time_offset).strftime("%Y-%m-%d %H:%M:%S")
                    
                    prob = float(np.clip(0.65 + np.random.normal(0, 0.1), 0, 1))
                    pred = 1 if prob >= 0.5 else 0
                    
                    cursor.execute("""
                        INSERT INTO api_logs (timestamp, antibody_id, vh, vl, tm_app, psr, prediction, probability, latency_ms, status_code)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 200)
                    """, (
                        timestamp, f"Drift-Clone-{i}", row["vh"], row["vl"], 
                        drifted_tm, drifted_psr, pred, prob, latency
                    ))
                
                # Seed 2 error logs for operational realism
                cursor.execute("""
                    INSERT INTO api_logs (timestamp, antibody_id, vh, vl, tm_app, psr, prediction, probability, latency_ms, status_code, error_message)
                    VALUES (?, 'Err-Clone', 'MOCK', 'MOCK', NULL, NULL, NULL, NULL, 5.0, 500, 'Internal Server Error: Sequence contains invalid amino acids')
                """, ((now - datetime.timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),))
                
                conn.commit()
                st.success("Seeding completed successfully! Refreshing dashboard...")
        conn.close()
    except Exception as e:
        st.error(f"Error seeding database: {e}")

# Ensure database exists and is seeded
seed_database_if_empty()

# -----------------------------------------------------------------------------
# App Layout & Header
# -----------------------------------------------------------------------------
st.title("🧬 Antibody Scoring & Production Monitoring Dashboard")
st.markdown("""
This dashboard monitors the live operational and statistical health of the deployed **ProtT5 + Logistic Regression** scoring model.
It tracks latencies, predictions, physical parameters, and automatically computes **Data Drift** to identify if production data has shifted away from training reference data.
""")

# Load logs
try:
    conn = get_db_connection()
    logs_df = pd.read_sql_query("SELECT * FROM api_logs ORDER BY timestamp DESC", conn)
    conn.close()
except Exception as e:
    st.error(f"Failed to read production logs database: {e}")
    logs_df = pd.DataFrame()

if logs_df.empty:
    st.warning("No production logs found in database.")
    st.stop()

# Convert types
logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/144/double-helix.png", width=100)
st.sidebar.title("Operational Limits")
latency_threshold = st.sidebar.slider("Latency Alert Threshold (ms)", 5, 100, 20)
st.sidebar.markdown("""
### Model Information
- **Algorithm**: Logistic Regression
- **Embeddings**: ProtT5 (1024 VH + 1024 VL dimensions)
- **Primary Task**: Binary risk classification
- **Positive Class**: High HIC developability risk (Retention >= 11.5 min)
""")

# -----------------------------------------------------------------------------
# KPI Row
# -----------------------------------------------------------------------------
st.header("📊 Key Operational & Performance Metrics")

total_calls = len(logs_df)
success_calls = len(logs_df[logs_df["status_code"] == 200])
failed_calls = len(logs_df[logs_df["status_code"] != 200])
error_rate = (failed_calls / total_calls) * 100.0 if total_calls > 0 else 0.0

valid_logs = logs_df[logs_df["status_code"] == 200]
avg_latency = valid_logs["latency_ms"].mean() if not valid_logs.empty else 0.0
p95_latency = valid_logs["latency_ms"].quantile(0.95) if not valid_logs.empty else 0.0
risk_prevalence = (valid_logs["prediction"].mean() * 100.0) if not valid_logs.empty else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total API Requests", f"{total_calls:,}", help="Total scoring requests received by FastAPI")
col2.metric("Average Latency", f"{avg_latency:.2f} ms", help="Mean response time of successful requests")
col3.metric("95th Percentile Latency", f"{p95_latency:.2f} ms", help="95% of calls were resolved within this time")
col4.metric("Error Rate", f"{error_rate:.2f} %", delta=f"{failed_calls} failures", delta_color="inverse")
col5.metric("Predicted High HIC Risk", f"{risk_prevalence:.1f} %", help="Percentage of inputs predicted as High HIC risk")

# -----------------------------------------------------------------------------
# Graphical Analytics Row
# -----------------------------------------------------------------------------
st.markdown("---")
c1, col_chart2 = st.columns(2)

with c1:
    st.subheader("⏱️ API Latency Trend over Time")
    if not valid_logs.empty:
        fig_lat = px.line(
            valid_logs.sort_values("timestamp"),
            x="timestamp",
            y="latency_ms",
            title="Inference Latency Timeseries (ms)",
            labels={"timestamp": "Time", "latency_ms": "Latency (ms)"},
            line_shape="linear",
            render_mode="svg"
        )
        # Highlight latency alert threshold
        fig_lat.add_hline(y=latency_threshold, line_dash="dash", line_color="red", annotation_text="Alert SLA")
        st.plotly_chart(fig_lat, use_container_width=True)
    else:
        st.write("No successful logs to display latency trends.")

with col_chart2:
    st.subheader("🎯 Risk Score Probability Distribution")
    if not valid_logs.empty:
        fig_score = px.histogram(
            valid_logs,
            x="probability",
            nbins=20,
            title="Distribution of Predicted Probabilities (Risk Class 1)",
            labels={"probability": "Predicted Probability (High HIC)"},
            color_discrete_sequence=["#FF4B4B"]
        )
        fig_score.add_vline(x=0.5, line_dash="dash", line_color="black", annotation_text="Decision Boundary (0.5)")
        st.plotly_chart(fig_score, use_container_width=True)
    else:
        st.write("No score probabilities available.")

# -----------------------------------------------------------------------------
# Data Drift Analysis (Evidently AI Integration)
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("🚨 Automated Data Drift Analysis (Evidently AI)")

if os.path.exists(dataset_path):
    ref_full = pd.read_parquet(dataset_path)
    ref_data = ref_full[["tm_app", "psr"]].copy()
    
    # Get current production features
    prod_data = valid_logs[["tm_app", "psr"]].dropna().copy()
    
    if len(prod_data) >= 5:
        with st.spinner("Calculating statistical drift scores..."):
            # Run Evidently report
            report = Report(metrics=[DataDriftPreset()])
            snapshot = report.run(reference_data=ref_data, current_data=prod_data)
            report_dict = snapshot.dict()
            
            # Extract results (Evidently v0.7 format)
            count_metric = report_dict["metrics"][0]
            drift_share_threshold = count_metric["config"]["drift_share"]
            drift_share = count_metric["value"]["share"]
            dataset_drifted = drift_share >= drift_share_threshold
            
            # Display overall summary
            if dataset_drifted:
                st.error("⚠️ DATA DRIFT DETECTED! The statistical distribution of incoming physical attributes has shifted significantly from the training reference data.")
            else:
                st.success("✅ NO DATA DRIFT DETECTED: The production data properties remain statistically aligned with training distributions.")
                
            # Details for each feature
            st.markdown("### Feature-by-Feature Statistical Drift Scores")
            
            # Build summary table
            features_summary = []
            for metric in report_dict["metrics"][1:]:
                col_name = metric["config"]["column"]
                p_value = metric["value"]
                threshold = metric["config"]["threshold"]
                method = metric["config"]["method"]
                drift_detected = p_value < threshold
                
                status_text = "🚨 Drifted" if drift_detected else "✅ Stable"
                features_summary.append({
                    "Feature": col_name,
                    "Drift Status": status_text,
                    "p-value": f"{p_value:.6f}",
                    "Threshold": f"{threshold:.2f}",
                    "Metric Score": f"{p_value:.4f}"
                })
                
            summary_df = pd.DataFrame(features_summary)
            st.dataframe(summary_df, use_container_width=True)
            
            # Add visualizations of reference vs current distribution
            st.markdown("### Visualizing Physical Feature Distributions")
            feat_choice = st.selectbox("Select Feature to Visualize Overlap", ["tm_app", "psr"])
            
            # Merge to plot overlap
            ref_data["Dataset"] = "Training Reference"
            prod_data["Dataset"] = "Live Production"
            combined_plot_df = pd.concat([ref_data[[feat_choice, "Dataset"]], prod_data[[feat_choice, "Dataset"]]])
            
            fig_overlap = px.histogram(
                combined_plot_df,
                x=feat_choice,
                color="Dataset",
                barmode="overlay",
                nbins=30,
                marginal="box",
                title=f"Distribution Overlap: Training vs Production ({feat_choice})",
                color_discrete_map={"Training Reference": "#1F77B4", "Live Production": "#FF7F0E"}
            )
            st.plotly_chart(fig_overlap, use_container_width=True)
    else:
        st.warning("Insufficient successful logs (minimum 5 needed) to calculate data drift.")
else:
    st.error("Reference dataset final_dataset.parquet not found. Unable to calculate drift.")

# -----------------------------------------------------------------------------
# Live Logs Viewer
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("📋 Live Production Logs (SQLite Database)")
st.markdown("Showing the latest requests logged directly from the FastAPI application. Useful for audit trails and screenshots.")
st.dataframe(logs_df, use_container_width=True)
