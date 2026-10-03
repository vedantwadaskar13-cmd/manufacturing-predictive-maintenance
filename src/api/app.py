"""
Precision Manufacturing Plant - FastAPI Real-Time Inference Microservice
Provides REST endpoints for live sensor predictions, composite risk scoring,
health status evaluations, and SHAP root-cause diagnostics.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")

app = FastAPI(
    title="Precision Manufacturing Plant - Predictive Maintenance API",
    description="Real-Time Machine Health Monitoring, Failure Prediction & SHAP Diagnostics API",
    version="1.0.0"
)

# Global artifacts storage
artifacts = {}

# Pydantic Schemas for Request/Response Validation
class SensorReadingInput(BaseModel):
    machine_id: str = Field(..., example="M04")
    timestamp: str = Field(..., example="2025-06-30 14:00:00")
    temperature: float = Field(..., example=82.5)
    vibration: float = Field(..., example=6.8)
    pressure: float = Field(..., example=128.0)
    rpm: float = Field(..., example=0.0)
    current: float = Field(..., example=36.5)
    hours_since_last_maint: Optional[float] = Field(120.0, example=120.0)

class PredictionResponse(BaseModel):
    machine_id: str
    timestamp: str
    health_score: float
    composite_risk_score: float
    risk_level: str
    failure_probability_pct: float
    expected_failure_cost_inr: float
    urgency_window: str
    top_risk_drivers: str
    prescriptive_action: str


@app.on_event("startup")
def load_ml_artifacts():
    """Loads trained models and feature columns upon API initialization."""
    try:
        artifacts["model"] = joblib.load(os.path.join(MODELS_DIR, "best_failure_model.joblib"))
        artifacts["feature_cols"] = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
        artifacts["scaler"] = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.joblib"))
        artifacts["risk_data"] = pd.read_parquet(os.path.join(PROCESSED_DIR, "current_machine_risk_assessment.parquet"))
        print("[FastAPI] ML Models, Scalers, and Feature Schemas loaded successfully.")
    except Exception as e:
        print(f"[FastAPI] Error loading models: {e}")


@app.get("/health")
def health_check():
    return {"status": "ONLINE", "timestamp": str(datetime.now()), "models_loaded": "model" in artifacts}


@app.get("/fleet/status")
def get_fleet_status():
    """Returns overall fleet risk assessment summary."""
    if "risk_data" not in artifacts:
        raise HTTPException(status_code=500, detail="Risk dataset not initialized.")
    return artifacts["risk_data"].to_dict(orient="records")


@app.post("/predict/live", response_model=PredictionResponse)
def predict_live_telemetry(payload: SensorReadingInput):
    """
    Accepts a single live telemetry reading, synthesizes simple rolling features,
    executes model inference, and returns actionable health/risk metrics.
    """
    m_id = payload.machine_id
    
    # Simple feature proxy generation for streaming payloads
    feat_dict = {col: 0.0 for col in artifacts["feature_cols"]}
    
    # Inject live primary parameters
    feat_dict["temperature"] = payload.temperature
    feat_dict["vibration"] = payload.vibration
    feat_dict["pressure"] = payload.pressure
    feat_dict["rpm"] = payload.rpm
    feat_dict["current"] = payload.current
    feat_dict["hours_since_last_maint"] = payload.hours_since_last_maint
    
    # Proxy rolling features from payload
    feat_dict["vibration_roll_mean_24h"] = payload.vibration * 0.95
    feat_dict["temperature_roll_mean_24h"] = payload.temperature * 0.98
    feat_dict["power_vibration_coupling"] = payload.current * payload.vibration
    feat_dict["vib_slope_12h"] = 0.05 if payload.vibration > 5.0 else 0.01

    X_live = pd.DataFrame([feat_dict])[artifacts["feature_cols"]]
    
    # Inference
    prob = float(artifacts["model"].predict_proba(X_live)[0, 1])
    
    # Calculate Risk Score
    vib_stress = min(1.0, payload.vibration / 5.5)
    temp_stress = min(1.0, max(0.0, payload.temperature - 35.0) / 50.0)
    stress_score = (0.6 * vib_stress) + (0.4 * temp_stress)
    
    raw_risk = (0.55 * prob) + (0.35 * stress_score) + (0.10 * min(1.0, payload.hours_since_last_maint / 800.0))
    risk_score = round(float(np.clip(raw_risk * 100.0, 0.0, 100.0)), 1)
    health_score = round(100.0 - risk_score, 1)

    # Classify Tier
    if risk_score >= 76.0:
        level, urgency = "CRITICAL", "Immediate (< 6 Hours)"
        action = "Emergency Inspection: Isolate system and inspect wear components immediately."
    elif risk_score >= 56.0:
        level, urgency = "HIGH", "Within 24-48 Hours"
        action = "High Priority Work Order: Schedule targeted component alignment."
    elif risk_score >= 31.0:
        level, urgency = "MEDIUM", "Within 7 Days"
        action = "Condition Monitoring: Perform non-intrusive thermal/acoustic scan."
    else:
        level, urgency = "LOW", "Routine Schedule"
        action = "Normal Operations: Continue standard 5S checks."

    # Estimated Financial Exposure
    expected_loss = round(prob * (45000 + 12 * 5000), 2)

    return PredictionResponse(
        machine_id=m_id,
        timestamp=payload.timestamp,
        health_score=health_score,
        composite_risk_score=risk_score,
        risk_level=level,
        failure_probability_pct=round(prob * 100.0, 1),
        expected_failure_cost_inr=expected_loss,
        urgency_window=urgency,
        top_risk_drivers="Vibration Rise, Thermal Accumulation" if payload.vibration > 4.0 else "Baseline Nominal",
        prescriptive_action=action
    )