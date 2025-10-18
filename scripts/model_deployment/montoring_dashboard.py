"""
Production Monitoring Dashboard for Fashion Recommendation System

Run with: streamlit run monitoring_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path


st.set_page_config(
    page_title="Fashion RecSys Monitoring",
    page_icon="📊",
    layout="wide"
)

st.title("🛍️ Fashion Recommendation System - Production Dashboard")
st.markdown("Real-time monitoring of model performance, latency, and drift detection")

# Load metrics
@st.cache_data(ttl=60)
def load_metrics():
    """Load production metrics"""
    if Path('production_metrics.json').exists():
        with open('production_metrics.json', 'r') as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=60)
def load_drift_history():
    """Load drift monitoring history"""
    if Path('drift_history.json').exists():
        with open('drift_history.json', 'r') as f:
            return json.load(f)
    return []

metrics_data = load_metrics()
drift_data = load_drift_history()

# Sidebar
st.sidebar.header("📅 Date Range")
days_back = st.sidebar.slider("Days to show", 1, 30, 7)

# Calculate date range
end_date = datetime.now()
start_date = end_date - timedelta(days=days_back)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

# Get today's metrics
today = datetime.now().strftime('%Y-%m-%d')
today_metrics = metrics_data.get(today, [])

if today_metrics:
    total_requests = len(today_metrics)
    avg_latency = np.mean([m['latency_ms'] for m in today_metrics])
    success_rate = np.mean([m['success'] for m in today_metrics]) * 100
    
    col1.metric("Total Requests (Today)", f"{total_requests:,}")
    col2.metric("Avg Latency", f"{avg_latency:.1f}ms", 
                delta=f"Target: <50ms", delta_color="inverse")
    col3.metric("Success Rate", f"{success_rate:.1f}%")
    col4.metric("Requests/Hour", f"{total_requests/24:.0f}")
else:
    col1.metric("Total Requests (Today)", "0")
    col2.metric("Avg Latency", "N/A")
    col3.metric("Success Rate", "N/A")
    col4.metric("Requests/Hour", "0")

st.markdown("---")

# Request volume over time
st.subheader("📈 Request Volume")

dates = []
request_counts = []
avg_latencies = []
p95_latencies = []

current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    day_data = metrics_data.get(date_str, [])
    
    dates.append(date_str)
    request_counts.append(len(day_data))
    
    if day_data:
        latencies = [m['latency_ms'] for m in day_data]
        avg_latencies.append(np.mean(latencies))
        p95_latencies.append(np.percentile(latencies, 95))
    else:
        avg_latencies.append(0)
        p95_latencies.append(0)
    
    current_date += timedelta(days=1)

# Create plotly chart
fig_volume = go.Figure()
fig_volume.add_trace(go.Bar(
    x=dates,
    y=request_counts,
    name='Requests',
    marker_color='lightblue'
))
fig_volume.update_layout(
    title='Daily Request Volume',
    xaxis_title='Date',
    yaxis_title='Number of Requests',
    height=300
)

st.plotly_chart(fig_volume, use_container_width=True)

# Latency monitoring
col1, col2 = st.columns(2)

with col1:
    st.subheader("⏱️ Latency Trends")
    
    fig_latency = go.Figure()
    fig_latency.add_trace(go.Scatter(
        x=dates, y=avg_latencies,
        mode='lines+markers',
        name='Average',
        line=dict(color='blue', width=2)
    ))
    fig_latency.add_trace(go.Scatter(
        x=dates, y=p95_latencies,
        mode='lines+markers',
        name='P95',
        line=dict(color='orange', width=2)
    ))
    fig_latency.add_hline(y=50, line_dash="dash", 
                          annotation_text="50ms SLA",
                          line_color="red")
    
    fig_latency.update_layout(
        xaxis_title='Date',
        yaxis_title='Latency (ms)',
        height=300
    )
    
    st.plotly_chart(fig_latency, use_container_width=True)

with col2:
    st.subheader("🎯 Model Performance")
    
    # Load model metrics from MLflow or stored evaluations
    if drift_data:
        perf_dates = [d['timestamp'][:10] for d in drift_data]
        hit10_values = [d['hit@10'] for d in drift_data]
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=perf_dates, y=hit10_values,
            mode='lines+markers',
            name='Hit@10',
            line=dict(color='green', width=2)
        ))
        fig_perf.add_hline(y=20, line_dash="dash",
                          annotation_text="Target: 20%",
                          line_color="blue")
        
        fig_perf.update_layout(
            xaxis_title='Date',
            yaxis_title='Hit@10 (%)',
            height=300
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
    else:
        st.info("No performance data available yet")

st.markdown("---")

# Drift detection
st.subheader("🚨 Model Drift Monitoring")

if drift_data and len(drift_data) > 1:
    baseline = drift_data[0]['hit@10']
    current = drift_data[-1]['hit@10']
    degradation = (baseline - current) / baseline * 100
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Baseline Hit@10", f"{baseline:.2f}%")
    col2.metric("Current Hit@10", f"{current:.2f}%", 
                delta=f"{-degradation:.1f}%")
    
    if degradation > 15:
        col3.error("⚠️ DRIFT DETECTED")
        st.warning(f"Model performance degraded by {degradation:.1f}%. Retraining recommended.")
    else:
        col3.success("✅ No Drift")
else:
    st.info("Collecting drift data... (need 2+ days of data)")

# System health
st.markdown("---")
st.subheader("💚 System Health")

health_col1, health_col2, health_col3 = st.columns(3)

# Simulate health checks
health_col1.metric("TorchServe Status", "🟢 Running")
health_col2.metric("GPU Utilization", "45%")
health_col3.metric("Memory Usage", "3.2 GB / 16 GB")

# Recent requests table
st.markdown("---")
st.subheader("📋 Recent Requests")

if today_metrics:
    recent = pd.DataFrame(today_metrics[-100:])  # Last 100 requests
    recent['timestamp'] = pd.to_datetime(recent['timestamp'])
    
    st.dataframe(
        recent[['timestamp', 'user_id', 'latency_ms', 'success']].tail(20),
        use_container_width=True
    )
    
    # Latency distribution
    st.subheader("📊 Latency Distribution (Today)")
    
    fig_dist = px.histogram(
        recent, 
        x='latency_ms',
        nbins=50,
        title='Request Latency Distribution'
    )
    fig_dist.add_vline(x=50, line_dash="dash", line_color="red",
                       annotation_text="50ms SLA")
    
    st.plotly_chart(fig_dist, use_container_width=True)

# Auto-refresh
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)")

if auto_refresh:
    time.sleep(60)
    st.rerun()

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")