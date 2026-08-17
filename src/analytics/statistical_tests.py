"""
Precision Manufacturing Plant - Statistical Testing & Hypothesis Verification
Performs two-sample t-tests, Mann-Whitney U tests, and correlation analysis
to statistically confirm physical degradation precursors.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def tag_pre_failure_windows(
    df_sensors: pd.DataFrame, 
    df_failures: pd.DataFrame, 
    window_hours: int = 24
) -> pd.DataFrame:
    """
    Labels each telemetry observation as either:
    1 = Within 'window_hours' before a catastrophic failure (Pre-Failure)
    0 = Normal Stable Operation
    """
    df = df_sensors.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_fail = df_failures.copy()
    df_fail["failure_timestamp"] = pd.to_datetime(df_fail["failure_timestamp"])
    
    df["is_pre_failure"] = 0
    
    for _, fail in df_fail.iterrows():
        m_id = fail["machine_id"]
        t_fail = fail["failure_timestamp"]
        t_start = t_fail - pd.Timedelta(hours=window_hours)
        
        mask = (df["machine_id"] == m_id) & (df["timestamp"] >= t_start) & (df["timestamp"] <= t_fail)
        df.loc[mask, "is_pre_failure"] = 1
        
    return df


def test_sensor_degradation_hypothesis(
    df_tagged: pd.DataFrame, 
    machine_id: str, 
    sensor_col: str, 
    alternative: str = "greater"
) -> Dict[str, float]:
    """
    Executes Welch's t-test and Mann-Whitney U test on a sensor stream 
    comparing Normal vs Pre-Failure operational distributions.
    """
    subset = df_tagged[df_tagged["machine_id"] == machine_id]
    normal_data = subset[subset["is_pre_failure"] == 0][sensor_col].dropna()
    pre_fail_data = subset[subset["is_pre_failure"] == 1][sensor_col].dropna()
    
    if len(normal_data) == 0 or len(pre_fail_data) == 0:
        raise ValueError(f"Insufficient data for machine {machine_id} on {sensor_col}")
        
    # 1. Normality check (Shapiro-Wilk on sample)
    sample_norm = normal_data.sample(min(500, len(normal_data)), random_state=42)
    _, p_shapiro = stats.shapiro(sample_norm)
    
    # 2. Welch's t-test (Does not assume equal variance)
    t_stat, p_ttest = stats.ttest_ind(pre_fail_data, normal_data, equal_var=False, alternative=alternative)
    
    # 3. Mann-Whitney U Non-Parametric Test
    u_stat, p_mwu = stats.mannwhitneyu(pre_fail_data, normal_data, alternative=alternative)
    
    # 4. Effect Size (Cohen's d)
    mean_diff = pre_fail_data.mean() - normal_data.mean()
    pooled_std = np.sqrt((normal_data.std()**2 + pre_fail_data.std()**2) / 2)
    cohens_d = mean_diff / (pooled_std + 1e-8)
    
    return {
        "machine_id": machine_id,
        "sensor": sensor_col,
        "normal_mean": round(float(normal_data.mean()), 3),
        "pre_fail_mean": round(float(pre_fail_data.mean()), 3),
        "mean_delta_pct": round(float((mean_diff / normal_data.mean()) * 100), 2),
        "cohens_d": round(float(cohens_d), 3),
        "welch_t_pval": float(p_ttest),
        "mann_whitney_pval": float(p_mwu),
        "is_significant_p01": (p_mwu < 0.01)
    }


def compute_production_quality_correlations(
    df_prod: pd.DataFrame, 
    df_sensors: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates rank correlations between telemetry drift and defect yield.
    """
    df_prod["timestamp"] = pd.to_datetime(df_prod["timestamp"])
    df_sensors["timestamp"] = pd.to_datetime(df_sensors["timestamp"])
    
    # Resample sensors to 8-hour shift windows matching production logs
    sensor_shift = df_sensors.groupby(["machine_id", pd.Grouper(key="timestamp", freq="8h")]).agg({
        "temperature": "mean",
        "vibration": "mean",
        "pressure": "mean",
        "current": "mean"
    }).reset_index()
    
    merged = pd.merge(df_prod, sensor_shift, on=["machine_id", "timestamp"], how="inner")
    
    results = []
    for m_id, group in merged.groupby("machine_id"):
        corr_vib, p_vib = stats.spearmanr(group["vibration"], group["defect_quantity"])
        corr_temp, p_temp = stats.spearmanr(group["temperature"], group["defect_quantity"])
        results.append({
            "machine_id": m_id,
            "spearman_vib_defects": round(float(corr_vib), 3),
            "p_val_vib": float(p_vib),
            "spearman_temp_defects": round(float(corr_temp), 3),
            "p_val_temp": float(p_temp)
        })
        
    return pd.DataFrame(results)