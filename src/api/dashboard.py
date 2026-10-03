"""
Precision Manufacturing Plant - Real-Time Operator Dashboard
Streamlit application simulating streaming IoT telemetry and visualizing live risk assessments.
"""

import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

# Configuration
API_URL = "http://127.0.0.1:8000/predict/live"

st.set_page_config(
    page_title="Precision Plant - Live Monitoring",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Precision Manufacturing Plant - Real-Time Telemetry Stream")
st.markdown("### Live Condition Monitoring, Risk Scoring & Prescriptive Dispatch")

# Sidebar Controls
st.sidebar.header("🕹️ Simulation Stream Controls")
selected_machine = st.sidebar.selectbox("Select Target Machine:", ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10"], index=3)
injection_mode = st.sidebar.radio("Simulation State:", ["Normal Nominal Running", "Gradual Degradation (Wear)", "Sudden Mechanical Fault Spike"])
sim_speed = st.sidebar.slider("Stream Interval (Seconds):", 1, 5, 2)

run_simulation = st.sidebar.checkbox("Start Live Streaming", value=False)

# Placeholders for dynamic UI elements
col1, col2, col3, col4 = st.columns(4)
health_metric = col1.empty()
risk_metric = col2.empty()
prob_metric = col3.empty()
cost_metric = col4.empty()

st.divider()

chart_col1, chart_col2 = st.columns(2)
vib_chart = chart_col1.empty()
temp_chart = chart_col2.empty()

alert_banner = st.empty()

# Initialize session state history for live streaming plots
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["timestamp", "vibration", "temperature", "pressure", "risk_score"])

if run_simulation:
    step = 0
    while run_simulation:
        step += 1
        curr_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # Synthesize physical telemetry based on simulation mode
        if injection_mode == "Normal Nominal Running":
            vib = float(np.round(1.2 + np.random.normal(0, 0.1), 2))
            temp = float(np.round(45.0 + np.random.normal(0, 0.5), 1))
            press = 135.0
        elif injection_mode == "Gradual Degradation (Wear)":
            vib = float(np.round(1.5 + (step * 0.25) + np.random.normal(0, 0.15), 2))
            temp = float(np.round(48.0 + (step * 0.8) + np.random.normal(0, 0.6), 1))
            press = float(np.round(135.0 - (step * 0.6), 1))
        else:  # Sudden Mechanical Fault
            vib = float(np.round(6.5 + np.random.normal(0, 0.3), 2))
            temp = float(np.round(82.0 + np.random.normal(0, 1.2), 1))
            press = 110.0

        payload = {
            "machine_id": selected_machine,
            "timestamp": curr_time,
            "temperature": temp,
            "vibration": vib,
            "pressure": max(0.0, press),
            "rpm": 1800.0,
            "current": 28.0,
            "hours_since_last_maint": 450.0 + step
        }

        # Send telemetry to API (or execute inline fallback if API offline)
        try:
            res = requests.post(API_URL, json=payload, timeout=2)
            data = res.json()
        except Exception:
            # Fallback inline calculation if API server is not running
            prob = min(0.95, (vib / 8.0))
            data = {
                "health_score": round(100 - (prob * 100), 1),
                "composite_risk_score": round(prob * 100, 1),
                "risk_level": "CRITICAL" if prob > 0.7 else ("HIGH" if prob > 0.4 else "LOW"),
                "failure_probability_pct": round(prob * 100, 1),
                "expected_failure_cost_inr": round(prob * 85000, 2),
                "urgency_window": "Immediate (< 6 Hours)" if prob > 0.7 else "Routine Schedule",
                "prescriptive_action": "Inspect main cylinder piston seals and bearings immediately."
            }

        # Update Top KPI Cards
        health_metric.metric("Asset Health Score", f"{data['health_score']}/100", delta=f"{'-' if data['health_score'] < 50 else '+'}{abs(100 - data['health_score']):.1f}")
        risk_metric.metric("Composite Risk Score", f"{data['composite_risk_score']}/100", delta=data['risk_level'], delta_color="inverse")
        prob_metric.metric("Failure Probability (24h)", f"{data['failure_probability_pct']}%")
        cost_metric.metric("Expected Financial Loss", f"₹{data['expected_failure_cost_inr']:,.2f}")

        # Update Stream History
        new_row = pd.DataFrame([{
            "timestamp": curr_time,
            "vibration": vib,
            "temperature": temp,
            "pressure": press,
            "risk_score": data["composite_risk_score"]
        }])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True).tail(30)

        # Plot Live Charts
        vib_chart.line_chart(st.session_state.history.set_index("timestamp")[["vibration"]], height=250)
        temp_chart.line_chart(st.session_state.history.set_index("timestamp")[["temperature"]], height=250)

        # Alert Banner Formatting
        if data["risk_level"] in ["HIGH", "CRITICAL"]:
            alert_banner.error(
                f"🚨 **{data['risk_level']} ALERT ON {selected_machine}** | Urgency: **{data['urgency_window']}**\n\n"
                f"**Prescriptive Dispatch Action:** {data['prescriptive_action']}"
            )
        else:
            alert_banner.success(f"✅ **{selected_machine} Operating Normally** | Telemetry parameters inside nominal boundaries.")

        time.sleep(sim_speed)
else:
    st.info("👈 Check 'Start Live Streaming' in the sidebar to launch the real-time simulation stream.")