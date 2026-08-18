"""
Precision Manufacturing Plant - Power BI Star Schema Data Exporter
Consolidates raw facts, aggregated metrics, and ML/XAI inferences into
dedicated dimension and fact tables for Power BI Desktop ingestion.
"""

import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
PBI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "powerbi", "data_model")
os.makedirs(PBI_DIR, exist_ok=True)


def export_star_schema():
    print("[1/5] Ingesting Cleaned Data and Model Predictions...")
    df_machines = pd.read_csv(os.path.join(PROCESSED_DIR, "machines.csv"))
    df_operators = pd.read_csv(os.path.join(PROCESSED_DIR, "operators.csv"))
    df_sensors = pd.read_parquet(os.path.join(PROCESSED_DIR, "sensor_telemetry.parquet"))
    df_prod = pd.read_parquet(os.path.join(PROCESSED_DIR, "production_logs.parquet"))
    df_down = pd.read_csv(os.path.join(PROCESSED_DIR, "downtime_records.csv"))
    df_fail = pd.read_csv(os.path.join(PROCESSED_DIR, "failure_events.csv"))
    df_risk = pd.read_parquet(os.path.join(PROCESSED_DIR, "current_machine_risk_assessment.parquet"))
    df_diag = pd.read_parquet(os.path.join(PROCESSED_DIR, "machine_shap_diagnostics.parquet"))

    print("[2/5] Generating Continuous Dimension: dim_date...")
    date_range = pd.date_range(start="2025-01-01", end="2026-12-31", freq="D")
    dim_date = pd.DataFrame({
        "Date": date_range.strftime("%Y-%m-%d"),
        "Year": date_range.year,
        "Month": date_range.strftime("%B"),
        "MonthNumber": date_range.month,
        "Quarter": "Q" + date_range.quarter.astype(str),
        "DayOfWeek": date_range.strftime("%A"),
        "IsWeekend": date_range.dayofweek.isin([5, 6]).astype(int)
    })

    print("[3/5] Consolidating Risk Assessment with SHAP Drivers...")
    # Harmonize join keys to string
    df_risk["machine_id"] = df_risk["machine_id"].astype(str)
    if "machine_id" not in df_diag.columns:
        df_diag["machine_id"] = df_diag.index.astype(str)
    else:
        df_diag["machine_id"] = df_diag["machine_id"].astype(str)

    # Dynamic driver column detection
    driver_col = "top_root_cause_drivers" if "top_root_cause_drivers" in df_diag.columns else (
        "primary_failure_driver" if "primary_failure_driver" in df_diag.columns else None
    )

    if driver_col:
        diag_subset = df_diag[["machine_id", driver_col]].drop_duplicates(subset=["machine_id"])
        fact_risk_xai = pd.merge(df_risk, diag_subset, on="machine_id", how="left")
        if driver_col != "top_root_cause_drivers":
            fact_risk_xai.rename(columns={driver_col: "top_root_cause_drivers"}, inplace=True)
    else:
        fact_risk_xai = df_risk.copy()
        fact_risk_xai["top_root_cause_drivers"] = "None"

    # Format date columns for Power Query compatibility
    df_sensors["Date"] = pd.to_datetime(df_sensors["timestamp"]).dt.strftime("%Y-%m-%d")
    df_prod["Date"] = pd.to_datetime(df_prod["timestamp"]).dt.strftime("%Y-%m-%d")
    df_down["Date"] = pd.to_datetime(df_down["start_time"]).dt.strftime("%Y-%m-%d")
    df_fail["Date"] = pd.to_datetime(df_fail["failure_timestamp"]).dt.strftime("%Y-%m-%d")

    print("[4/5] Serializing Model Tables to powerbi/data_model/...")
    dim_date.to_csv(os.path.join(PBI_DIR, "dim_date.csv"), index=False)
    df_machines.to_csv(os.path.join(PBI_DIR, "dim_machines.csv"), index=False)
    df_operators.to_csv(os.path.join(PBI_DIR, "dim_operators.csv"), index=False)
    df_sensors.to_csv(os.path.join(PBI_DIR, "fact_sensor_telemetry.csv"), index=False)
    df_prod.to_csv(os.path.join(PBI_DIR, "fact_production_logs.csv"), index=False)
    df_down.to_csv(os.path.join(PBI_DIR, "fact_downtime_events.csv"), index=False)
    df_fail.to_csv(os.path.join(PBI_DIR, "fact_failure_events.csv"), index=False)
    fact_risk_xai.to_csv(os.path.join(PBI_DIR, "fact_machine_risk_xai.csv"), index=False)

    print(f"\n[5/5] Export Complete. 8 CSV tables generated in: {PBI_DIR}")


if __name__ == "__main__":
    export_star_schema()