"""
Precision Manufacturing Plant - Predictive Maintenance Risk Engine
Computes Composite Risk Scores (0-100), Machine Health Scores (0-100),
Expected Financial Loss (₹), and Prescriptive Maintenance Actions.
"""

import os
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
FEATURES_DIR = os.path.join(PROCESSED_DIR, "features")

# Machine Specific Physical Limits and Replacement Costs
MACHINE_SPECS = {
    "M01": {"type": "CNC Lathe", "crit_vib": 4.5, "crit_temp": 75.0, "base_cost": 35000, "est_dt_hours": 10, "action": "Inspect spindle bearings, check chuck clamp pressure and toolholder balance"},
    "M02": {"type": "CNC Milling", "crit_vib": 5.0, "crit_temp": 80.0, "base_cost": 42000, "est_dt_hours": 12, "action": "Perform dynamic spindle balance, inspect linear guideways and drawbar tension"},
    "M03": {"type": "CNC Milling", "crit_vib": 5.2, "crit_temp": 82.0, "base_cost": 45000, "est_dt_hours": 12, "action": "Inspect spindle cartridge, check axis backlash and coolant delivery flow"},
    "M04": {"type": "Hydraulic Press", "crit_vib": 6.5, "crit_temp": 85.0, "base_cost": 58000, "est_dt_hours": 16, "action": "Inspect main cylinder piston seals, proportional relief valve, check for pressure surge"},
    "M05": {"type": "Hydraulic System", "crit_vib": 5.5, "crit_temp": 80.0, "base_cost": 29000, "est_dt_hours": 8, "action": "Inspect hydraulic pump impellers, flush line filters, check fluid contamination"},
    "M06": {"type": "Pump", "crit_vib": 3.5, "crit_temp": 65.0, "base_cost": 18000, "est_dt_hours": 6, "action": "Check mechanical seal leakage, inspect impeller for cavitation pitting and alignment"},
    "M07": {"type": "Pump", "crit_vib": 3.8, "crit_temp": 68.0, "base_cost": 22000, "est_dt_hours": 7, "action": "Replace shaft packing seals, check suction line strainer, measure casing vibration"},
    "M08": {"type": "Electric Motor", "crit_vib": 4.0, "crit_temp": 90.0, "base_cost": 48000, "est_dt_hours": 14, "action": "Inspect drive-end bearings, check stator insulation resistance (megger test), check phase current"},
    "M09": {"type": "Electric Motor", "crit_vib": 4.5, "crit_temp": 95.0, "base_cost": 55000, "est_dt_hours": 15, "action": "Re-grease bearings, inspect cooling fan shroud, verify thermal overload relays"},
    "M10": {"type": "Compressor", "crit_vib": 7.0, "crit_temp": 105.0, "base_cost": 65000, "est_dt_hours": 18, "action": "Inspect screw elements, check oil separator differential pressure, clean intercooler"}
}

HOURLY_DOWNTIME_RATE = 5000.0  # ₹5,000 / hour


class PredictiveMaintenanceRiskEngine:
    def __init__(self):
        self.model = joblib.load(os.path.join(MODELS_DIR, "best_failure_model.joblib"))
        self.feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
        self.optimal_threshold = joblib.load(os.path.join(MODELS_DIR, "optimal_threshold.joblib"))

    def compute_telemetry_stress(self, row: pd.Series, spec: dict) -> float:
        """Calculates normalized physical stress index based on sensor deviations."""
        vib_stress = min(1.0, row["vibration_roll_mean_24h"] / spec["crit_vib"])
        temp_stress = min(1.0, max(0.0, row["temperature_roll_mean_24h"] - 35.0) / (spec["crit_temp"] - 35.0))
        
        # Pressure stress (for hydraulic/compressor systems)
        if spec["type"] in ["Hydraulic Press", "Hydraulic System", "Compressor"]:
            press_stress = min(1.0, row["press_rolling_std_12h"] / 8.0) if "press_rolling_std_12h" in row else 0.2
        else:
            press_stress = 0.0

        stress = (0.55 * vib_stress) + (0.30 * temp_stress) + (0.15 * press_stress)
        return float(np.clip(stress, 0.0, 1.0))

    def evaluate_machine_state(self, machine_features: pd.DataFrame) -> pd.DataFrame:
        """
        Takes the latest feature vector for all machines and produces
        actionable health, risk, and prescriptive maintenance outputs.
        """
        # Ensure latest record per machine
        latest_df = machine_features.sort_values("timestamp").groupby("machine_id").last().reset_index()

        X = latest_df[self.feature_cols]
        failure_probs = self.model.predict_proba(X)[:, 1]

        assessment_results = []

        for idx, row in latest_df.iterrows():
            m_id = row["machine_id"]
            spec = MACHINE_SPECS.get(m_id, MACHINE_SPECS["M01"])
            prob = float(failure_probs[idx])

            # 1. Telemetry Stress Score (0.0 to 1.0)
            stress_score = self.compute_telemetry_stress(row, spec)

            # 2. Aging / Recency Factor (0.0 to 1.0)
            hrs_since_maint = float(row.get("hours_since_last_maint", 100.0))
            age_factor = min(1.0, hrs_since_maint / (35 * 24.0))

            # 3. Composite Risk Score Calculation (0 to 100)
            raw_risk = (0.55 * prob) + (0.30 * stress_score) + (0.15 * age_factor)
            risk_score = round(float(np.clip(raw_risk * 100.0, 0.0, 100.0)), 1)
            health_score = round(100.0 - risk_score, 1)

            # 4. Categorize Risk Tier & Prescriptive Window
            if risk_score >= 76.0 or prob >= 0.70:
                risk_level = "CRITICAL"
                urgency = "Immediate (< 6 Hours)"
                action_status = "Emergency Work Order: Isolate machine and execute targeted overhaul"
            elif risk_score >= 56.0 or prob >= self.optimal_threshold:
                risk_level = "HIGH"
                urgency = "Within 24-48 Hours"
                action_status = "High-Priority Maintenance: Schedule planned component replacement"
            elif risk_score >= 31.0:
                risk_level = "MEDIUM"
                urgency = "Within 7 Days"
                action_status = "Condition Monitoring: Conduct vibration/thermal inspection"
            else:
                risk_level = "LOW"
                urgency = "Routine Next Cycle"
                action_status = "Normal Operation: Continue regular 5S and lubrication checks"

            # 5. Financial Loss Calculation
            total_fail_cost = spec["base_cost"] + (spec["est_dt_hours"] * HOURLY_DOWNTIME_RATE)
            expected_loss = round(prob * total_fail_cost, 2)

            # 6. Primary Risk Factor Drivers
            drivers = []
            if row["vibration"] > (spec["crit_vib"] * 0.75):
                drivers.append("High Vibration Amplitude")
            if row["vib_slope_12h"] > 0.04:
                drivers.append("Accelerating Vibration Trend")
            if row["temperature"] > (spec["crit_temp"] * 0.80):
                drivers.append("Thermal Heat Accumulation")
            if hrs_since_maint > 600:
                drivers.append("Extended Operation Since Maintenance")
            if not drivers:
                drivers.append("Nominal Baseline Parameters")

            assessment_results.append({
                "machine_id": m_id,
                "machine_name": f"{spec['type']} ({m_id})",
                "machine_type": spec["type"],
                "last_timestamp": str(row["timestamp"]),
                "failure_probability_pct": round(prob * 100.0, 1),
                "composite_risk_score": risk_score,
                "health_score": health_score,
                "risk_level": risk_level,
                "urgency_window": urgency,
                "expected_failure_cost_inr": expected_loss,
                "total_breakdown_exposure_inr": total_fail_cost,
                "top_risk_drivers": ", ".join(drivers[:3]),
                "prescriptive_action": f"{spec['action']}. {action_status}",
                "temperature_current": round(float(row["temperature"]), 1),
                "vibration_current": round(float(row["vibration"]), 2),
                "pressure_current": round(float(row["pressure"]), 1),
                "hours_since_maintenance": round(hrs_since_maint, 1)
            })

        df_risk = pd.DataFrame(assessment_results).sort_values("composite_risk_score", ascending=False)
        return df_risk


def main():
    print("[1/3] Loading Latest Feature Stream for Machine Fleet...")
    features_all = pd.read_parquet(os.path.join(FEATURES_DIR, "features_all.parquet"))

    print("[2/3] Executing Predictive Maintenance Risk Engine...")
    engine = PredictiveMaintenanceRiskEngine()
    df_risk_summary = engine.evaluate_machine_state(features_all)

    print("[3/3] Exporting Risk Assessments to data/processed/...")
    output_path = os.path.join(PROCESSED_DIR, "current_machine_risk_assessment.parquet")
    df_risk_summary.to_parquet(output_path, index=False)
    df_risk_summary.to_csv(os.path.join(PROCESSED_DIR, "current_machine_risk_assessment.csv"), index=False)

    print("\nFleet Risk & Health Assessment Summary:")
    print("=" * 115)
    print(f"{'Machine':<8} | {'Type':<16} | {'Health':<7} | {'Risk Score':<10} | {'Risk Level':<9} | {'Fail Prob':<10} | {'Exp Loss (Rs.)':<14} | {'Urgency Window':<20}")
    print("-" * 115)
    for _, r in df_risk_summary.iterrows():
        print(f"{r['machine_id']:<8} | {r['machine_type']:<16} | {r['health_score']:>6.1f} | {r['composite_risk_score']:>10.1f} | {r['risk_level']:<9} | {r['failure_probability_pct']:>9.1f}% | Rs.{r['expected_failure_cost_inr']:>12,.2f} | {r['urgency_window']:<20}")
    print("=" * 115)


if __name__ == "__main__":
    main()