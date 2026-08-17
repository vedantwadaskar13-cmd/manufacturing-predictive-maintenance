"""
Precision Manufacturing Plant - Time-Series Feature Engineering Pipeline
Constructs causal rolling moments, thermodynamic drift metrics, cross-parameter couplings,
and chronological dataset splits without temporal leakage.
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
FEATURES_DIR = os.path.join(PROCESSED_DIR, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)


def compute_rolling_slopes(series: pd.Series, window: int = 12) -> pd.Series:
    """
    Computes rolling linear regression slope over backward-looking window.
    Slope = Cov(x, y) / Var(x)
    """
    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def calc_slope(y_window):
        if len(y_window) < window or np.isnan(y_window).any():
            return 0.0
        y_mean = y_window.mean()
        cov = ((x - x_mean) * (y_window - y_mean)).sum()
        return cov / x_var

    return series.rolling(window=window, min_periods=window).apply(calc_slope, raw=True).fillna(0.0)


def build_machine_features(
    df_sensors: pd.DataFrame,
    df_machines: pd.DataFrame,
    df_maint: pd.DataFrame,
    df_failures: pd.DataFrame,
    prediction_horizon_hours: int = 24
) -> pd.DataFrame:
    """
    Generates causal ML features for each machine stream.
    """
    df_sensors = df_sensors.copy().sort_values(["machine_id", "timestamp"])
    df_sensors["timestamp"] = pd.to_datetime(df_sensors["timestamp"])
    df_maint = df_maint.copy()
    df_maint["maintenance_date"] = pd.to_datetime(df_maint["maintenance_date"])
    df_failures = df_failures.copy()
    df_failures["failure_timestamp"] = pd.to_datetime(df_failures["failure_timestamp"])

    feature_frames = []

    for m_id, group in df_sensors.groupby("machine_id"):
        group = group.copy().sort_values("timestamp").reset_index(drop=True)
        m_meta = df_machines[df_machines["machine_id"] == m_id].iloc[0]
        base_power = float(m_meta["rated_power_kw"])

        # ---------------------------------------------------------
        # 1. Statistical Rolling Moments (Asymmetric Past-Looking)
        # ---------------------------------------------------------
        for col in ["vibration", "temperature", "pressure", "current"]:
            group[f"{col}_roll_mean_3h"] = group[col].rolling(3, min_periods=1).mean()
            group[f"{col}_roll_mean_12h"] = group[col].rolling(12, min_periods=1).mean()
            group[f"{col}_roll_mean_24h"] = group[col].rolling(24, min_periods=1).mean()

            group[f"{col}_roll_std_3h"] = group[col].rolling(3, min_periods=1).std().fillna(0.0)
            group[f"{col}_roll_std_12h"] = group[col].rolling(12, min_periods=1).std().fillna(0.0)
            group[f"{col}_roll_std_24h"] = group[col].rolling(24, min_periods=1).std().fillna(0.0)

            group[f"{col}_peak_to_avg_24h"] = (
                group[col] / (group[f"{col}_roll_mean_24h"] + 1e-6)
            ).clip(0.0, 10.0)

        # ---------------------------------------------------------
        # 2. Thermodynamic & Physical Signal Slopes
        # ---------------------------------------------------------
        group["temp_slope_12h"] = compute_rolling_slopes(group["temperature"], window=12)
        group["vib_slope_12h"] = compute_rolling_slopes(group["vibration"], window=12)
        group["press_slope_12h"] = compute_rolling_slopes(group["pressure"], window=12)

        # ---------------------------------------------------------
        # 3. Cross-Parameter Coupled Features
        # ---------------------------------------------------------
        # Power-Vibration Energy Intensity (Current * Vibration)
        group["power_vibration_coupling"] = group["current"] * group["vibration"]
        # Thermal-Load Coefficient (Temp / (Current + 1e-3))
        group["thermal_load_ratio"] = group["temperature"] / (group["current"] + 1e-3)

        # ---------------------------------------------------------
        # 4. Maintenance Recency & Cumulative Run Hours
        # ---------------------------------------------------------
        m_maints = df_maint[df_maint["machine_id"] == m_id]["maintenance_date"].sort_values().tolist()
        m_fails = df_failures[df_failures["machine_id"] == m_id]["failure_timestamp"].sort_values().tolist()

        hours_since_maint = []
        cum_failures = []

        for curr_time in group["timestamp"]:
            # Find past maintenance events strictly <= curr_time
            past_m = [m for m in m_maints if m <= curr_time]
            if past_m:
                delta_m = (curr_time - past_m[-1]).total_seconds() / 3600.0
            else:
                delta_m = 100.0  # Initial baseline offset
            hours_since_maint.append(delta_m)

            # Cumulative prior failures
            past_f = [f for f in m_fails if f < curr_time]
            cum_failures.append(len(past_f))

        group["hours_since_last_maint"] = hours_since_maint
        group["prior_failure_count"] = cum_failures

        # ---------------------------------------------------------
        # 5. Target Formulation: failure_next_24h (Forward-Looking Target Only)
        # ---------------------------------------------------------
        group["failure_next_24h"] = 0
        for f_time in m_fails:
            # Observation timestamp is in target window if: 0 < (f_time - timestamp) <= 24h
            target_mask = (group["timestamp"] < f_time) & (
                group["timestamp"] >= (f_time - pd.Timedelta(hours=prediction_horizon_hours))
            )
            group.loc[target_mask, "failure_next_24h"] = 1

        feature_frames.append(group)

    df_full_features = pd.concat(feature_frames, ignore_index=True)
    return df_full_features


def create_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.67,
    val_ratio: float = 0.165
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the feature dataset into Train, Validation, and Test sets
    strictly by chronological timestamps across all machines.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    unique_timestamps = np.sort(df["timestamp"].unique())
    n_times = len(unique_timestamps)

    train_end_idx = int(n_times * train_ratio)
    val_end_idx = int(n_times * (train_ratio + val_ratio))

    train_cutoff = unique_timestamps[train_end_idx]
    val_cutoff = unique_timestamps[val_end_idx]

    train_df = df[df["timestamp"] < train_cutoff].copy()
    val_df = df[(df["timestamp"] >= train_cutoff) & (df["timestamp"] < val_cutoff)].copy()
    test_df = df[df["timestamp"] >= val_cutoff].copy()

    return train_df, val_df, test_df


def main():
    print("[1/4] Loading Processed Tables for Feature Extraction...")
    df_sensors = pd.read_parquet(os.path.join(PROCESSED_DIR, "sensor_telemetry.parquet"))
    df_machines = pd.read_csv(os.path.join(PROCESSED_DIR, "machines.csv"))
    df_maint = pd.read_csv(os.path.join(PROCESSED_DIR, "maintenance_logs.csv"))
    df_failures = pd.read_csv(os.path.join(PROCESSED_DIR, "failure_events.csv"))

    print("[2/4] Engineering Asymmetric Rolling Moments & Physical Features...")
    df_features = build_machine_features(df_sensors, df_machines, df_maint, df_failures, prediction_horizon_hours=24)

    print("[3/4] Partitioning Data with Chronological Split...")
    train_df, val_df, test_df = create_chronological_splits(df_features)

    print("[4/4] Serializing Feature Matrices...")
    df_features.to_parquet(os.path.join(FEATURES_DIR, "features_all.parquet"), index=False)
    train_df.to_parquet(os.path.join(FEATURES_DIR, "features_train.parquet"), index=False)
    val_df.to_parquet(os.path.join(FEATURES_DIR, "features_val.parquet"), index=False)
    test_df.to_parquet(os.path.join(FEATURES_DIR, "features_test.parquet"), index=False)

    print("\nFeature Engineering Pipeline Summary:")
    print("----------------------------------------------------------------------")
    print(f"Total Feature Rows:           {len(df_features):>8,d} records")
    print(f"Total Feature Columns:        {df_features.shape[1]:>8d} features")
    print(f"Training Set (Months 1-4):    {len(train_df):>8,d} rows | Failures: {train_df['failure_next_24h'].sum():>4d} ({train_df['failure_next_24h'].mean()*100:.2f}%)")
    print(f"Validation Set (Month 5):     {len(val_df):>8,d} rows | Failures: {val_df['failure_next_24h'].sum():>4d} ({val_df['failure_next_24h'].mean()*100:.2f}%)")
    print(f"Test Set (Month 6):           {len(test_df):>8,d} rows | Failures: {test_df['failure_next_24h'].sum():>4d} ({test_df['failure_next_24h'].mean()*100:.2f}%)")
    print(f"Features Directory:           {FEATURES_DIR}")
    print("----------------------------------------------------------------------")


if __name__ == "__main__":
    main()