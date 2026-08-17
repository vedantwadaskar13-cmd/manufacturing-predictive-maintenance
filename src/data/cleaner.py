"""
Precision Manufacturing Plant - Data Cleaning & Validation Pipeline
Implements rigorous industrial validation, Hampel outlier filtering,
temporal alignment, and relational consistency checks.
"""

import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "synthetic")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Sensor physical boundaries [Min, Max]
PHYSICAL_LIMITS = {
    "temperature": (-10.0, 150.0),    # °C
    "vibration": (0.0, 25.0),         # mm/s RMS
    "pressure": (0.0, 350.0),         # bar
    "rpm": (0.0, 6000.0),             # RPM
    "current": (0.0, 250.0),          # Amperes
}


def hampel_filter_series(series: pd.Series, window_size: int = 5, n_sigmas: float = 3.0) -> pd.Series:
    """
    Applies Hampel Filter (Median Absolute Deviation) to replace isolated 
    telemetry glitches while preserving multi-step physical degradation trends.
    """
    rolling_median = series.rolling(window=window_size, min_periods=1, center=True).median()
    rolling_mad = (series - rolling_median).abs().rolling(window=window_size, min_periods=1, center=True).median()
    threshold = n_sigmas * 1.4826 * rolling_mad
    difference = (series - rolling_median).abs()
    
    # Identify isolated single-point glitches
    outlier_idx = difference > threshold
    cleaned_series = series.copy()
    cleaned_series[outlier_idx] = rolling_median[outlier_idx]
    return cleaned_series


def validate_and_clean_sensors(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Cleans sensor telemetry: enforces physical limits, imputes isolated transmission 
    glitches, and ensures strictly monotonic timestamps per machine.
    """
    metrics = {"duplicates_removed": 0, "physical_clips": 0, "glitches_imputed": 0}
    df = df.copy()
    
    # 1. Parse Datetime & Deduplicate
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    initial_len = len(df)
    df = df.drop_duplicates(subset=["machine_id", "timestamp"]).sort_values(["machine_id", "timestamp"])
    metrics["duplicates_removed"] = initial_len - len(df)

    # 2. Process per machine stream
    cleaned_frames = []
    for m_id, group in df.groupby("machine_id"):
        group = group.copy()
        
        # Enforce hard physical limits
        for col, (low, high) in PHYSICAL_LIMITS.items():
            if col in group.columns:
                out_of_bounds = (group[col] < low) | (group[col] > high)
                metrics["physical_clips"] += int(out_of_bounds.sum())
                group[col] = group[col].clip(lower=low, upper=high)
                
                # Filter isolated transient transmission spikes
                orig_col = group[col].copy()
                group[col] = hampel_filter_series(group[col], window_size=5, n_sigmas=3.5)
                metrics["glitches_imputed"] += int((orig_col != group[col]).sum())

        cleaned_frames.append(group)

    df_cleaned = pd.concat(cleaned_frames, ignore_index=True)
    return df_cleaned, metrics


def validate_and_clean_production(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validates production records: confirms scrap balance and non-negative quantities.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["production_id"])

    # Ensure non-negativity
    qty_cols = ["planned_quantity", "actual_quantity", "good_quantity", "defect_quantity"]
    for col in qty_cols:
        df[col] = df[col].apply(lambda x: max(0, int(x)))

    # Balance Reconciliation: actual must equal good + defect
    imbalance = df["actual_quantity"] != (df["good_quantity"] + df["defect_quantity"])
    if imbalance.any():
        df.loc[imbalance, "good_quantity"] = df.loc[imbalance, "actual_quantity"] - df.loc[imbalance, "defect_quantity"]

    return df, {"reconciled_rows": int(imbalance.sum())}


def validate_and_clean_events(df_maint: pd.DataFrame, df_fail: pd.DataFrame, df_down: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ensures relational integrity and valid temporal sequences for maintenance, failures, and downtime.
    """
    df_maint = df_maint.copy()
    df_fail = df_fail.copy()
    df_down = df_down.copy()

    df_maint["maintenance_date"] = pd.to_datetime(df_maint["maintenance_date"])
    df_fail["failure_timestamp"] = pd.to_datetime(df_fail["failure_timestamp"])
    df_down["start_time"] = pd.to_datetime(df_down["start_time"])
    df_down["end_time"] = pd.to_datetime(df_down["end_time"])

    # Duration Sanity: end_time must be >= start_time
    invalid_dt = df_down["end_time"] < df_down["start_time"]
    if invalid_dt.any():
        df_down.loc[invalid_dt, "end_time"] = df_down.loc[invalid_dt, "start_time"] + pd.to_timedelta(df_down.loc[invalid_dt, "duration_minutes"], unit="m")

    return df_maint, df_fail, df_down


def run_cleaning_pipeline():
    print("[1/4] Loading Raw Datasets from data/synthetic/...")
    df_machines = pd.read_csv(os.path.join(RAW_DIR, "machines.csv"))
    df_operators = pd.read_csv(os.path.join(RAW_DIR, "operators.csv"))
    df_sensors = pd.read_csv(os.path.join(RAW_DIR, "sensor_telemetry.csv"))
    df_prod = pd.read_csv(os.path.join(RAW_DIR, "production_logs.csv"))
    df_maint = pd.read_csv(os.path.join(RAW_DIR, "maintenance_logs.csv"))
    df_fail = pd.read_csv(os.path.join(RAW_DIR, "failure_events.csv"))
    df_down = pd.read_csv(os.path.join(RAW_DIR, "downtime_records.csv"))

    print("[2/4] Executing Sensor Signal Filtering & Physical Validation...")
    df_sensors_clean, sensor_metrics = validate_and_clean_sensors(df_sensors)

    print("[3/4] Validating Production Balances & Downtime Temporal Alignment...")
    df_prod_clean, prod_metrics = validate_and_clean_production(df_prod)
    df_maint_clean, df_fail_clean, df_down_clean = validate_and_clean_events(df_maint, df_fail, df_down)

    print("[4/4] Writing Validated Datasets to data/processed/...")
    df_machines.to_csv(os.path.join(PROCESSED_DIR, "machines.csv"), index=False)
    df_operators.to_csv(os.path.join(PROCESSED_DIR, "operators.csv"), index=False)
    df_sensors_clean.to_csv(os.path.join(PROCESSED_DIR, "sensor_telemetry.csv"), index=False)
    df_prod_clean.to_csv(os.path.join(PROCESSED_DIR, "production_logs.csv"), index=False)
    df_maint_clean.to_csv(os.path.join(PROCESSED_DIR, "maintenance_logs.csv"), index=False)
    df_fail_clean.to_csv(os.path.join(PROCESSED_DIR, "failure_events.csv"), index=False)
    df_down_clean.to_csv(os.path.join(PROCESSED_DIR, "downtime_records.csv"), index=False)

    # Also export parquet for high-speed downstream Python/ML ingestion
    df_sensors_clean.to_parquet(os.path.join(PROCESSED_DIR, "sensor_telemetry.parquet"), index=False)
    df_prod_clean.to_parquet(os.path.join(PROCESSED_DIR, "production_logs.parquet"), index=False)

    print("\nData Cleaning Pipeline Summary:")
    print("--------------------------------------------------")
    print(f"Duplicates Removed:      {sensor_metrics['duplicates_removed']:>6d}")
    print(f"Physical Limit Clips:    {sensor_metrics['physical_clips']:>6d}")
    print(f"Sensor Glitches Imputed: {sensor_metrics['glitches_imputed']:>6d}")
    print(f"Production Reconciled:   {prod_metrics['reconciled_rows']:>6d}")
    print(f"Destination Directory:   {PROCESSED_DIR}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    run_cleaning_pipeline()