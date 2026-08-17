"""
Precision Manufacturing Plant - KPI Analytics Engine
Calculates Availability, Performance, Quality, OEE, MTBF, MTTR,
Machine Utilization, and Financial Cost Losses.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
OUTPUT_KPI_DIR = os.path.join(PROCESSED_DIR, "kpis")
os.makedirs(OUTPUT_KPI_DIR, exist_ok=True)

# Financial Constants for Precision Manufacturing Plant
PLANT_HOURLY_REVENUE_LOSS = 5000.0  # ₹/hour of lost production
SCRAP_UNIT_COST = 350.0             # ₹/scrap part


def compute_fleet_oee(
    df_machines: pd.DataFrame,
    df_prod: pd.DataFrame,
    df_down: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes overall asset Availability, Performance, Quality, and OEE.
    """
    df_prod = df_prod.copy()
    df_down = df_down.copy()
    df_prod["timestamp"] = pd.to_datetime(df_prod["timestamp"])
    df_down["start_time"] = pd.to_datetime(df_down["start_time"])

    # 1. Total Planned Production Hours (6 Months = 180 Days * 24 Hours = 4,320 Hours)
    total_calendar_hours = 180 * 24.0

    # 2. Production Aggregations per Machine
    prod_summary = df_prod.groupby("machine_id").agg(
        total_planned_qty=("planned_quantity", "sum"),
        total_actual_qty=("actual_quantity", "sum"),
        total_good_qty=("good_quantity", "sum"),
        total_defect_qty=("defect_quantity", "sum"),
        avg_actual_cycle_time=("actual_cycle_time_sec", "mean")
    ).reset_index()

    # 3. Downtime Aggregations per Machine
    down_summary = df_down.groupby("machine_id").agg(
        total_downtime_minutes=("duration_minutes", "sum"),
        unplanned_downtime_minutes=("duration_minutes", lambda x: x[df_down.loc[x.index, "planned_flag"] == 0].sum()),
        planned_downtime_minutes=("duration_minutes", lambda x: x[df_down.loc[x.index, "planned_flag"] == 1].sum())
    ).reset_index()

    # Merge tables
    df_kpi = pd.merge(df_machines[["machine_id", "machine_name", "machine_type"]], prod_summary, on="machine_id", how="left")
    df_kpi = pd.merge(df_kpi, down_summary, on="machine_id", how="left").fillna(0)

    # Convert downtime to hours
    df_kpi["total_downtime_hours"] = df_kpi["total_downtime_minutes"] / 60.0
    df_kpi["unplanned_downtime_hours"] = df_kpi["unplanned_downtime_minutes"] / 60.0
    df_kpi["planned_downtime_hours"] = df_kpi["planned_downtime_minutes"] / 60.0

    # Operational Running Time
    df_kpi["operating_hours"] = total_calendar_hours - df_kpi["total_downtime_hours"]

    # Ideal Cycle Times: 45s for CNC, 30s for Others
    df_kpi["ideal_cycle_sec"] = df_kpi["machine_type"].apply(lambda x: 45.0 if "CNC" in x else 30.0)

    # --- OEE Elements Calculation ---
    # Availability = Operating Time / Planned Production Time
    df_kpi["availability_pct"] = np.clip(
        ((total_calendar_hours - df_kpi["total_downtime_hours"]) / total_calendar_hours) * 100.0, 0.0, 100.0
    )

    # Performance = (Ideal Cycle Time * Total Actual Qty) / (Operating Time in seconds)
    operating_seconds = df_kpi["operating_hours"] * 3600.0
    df_kpi["performance_pct"] = np.clip(
        ((df_kpi["ideal_cycle_sec"] * df_kpi["total_actual_qty"]) / operating_seconds) * 100.0, 0.0, 100.0
    )

    # Quality = Good Quantity / Total Actual Quantity
    df_kpi["quality_pct"] = np.clip(
        (df_kpi["total_good_qty"] / df_kpi["total_actual_qty"]) * 100.0, 0.0, 100.0
    )

    # Overall OEE
    df_kpi["oee_pct"] = (
        (df_kpi["availability_pct"] / 100.0) *
        (df_kpi["performance_pct"] / 100.0) *
        (df_kpi["quality_pct"] / 100.0)
    ) * 100.0

    # Utilization Rate (Operating Hours / Calendar Window)
    df_kpi["utilization_rate_pct"] = (df_kpi["operating_hours"] / total_calendar_hours) * 100.0

    return df_kpi.round(2)


def compute_reliability_and_financials(
    df_kpi: pd.DataFrame,
    df_failures: pd.DataFrame,
    df_maint: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes MTBF, MTTR, Failure Rates, and financial losses.
    """
    df_fail = df_failures.copy()
    df_maint = df_maint.copy()

    # Aggregate Failures
    fail_agg = df_fail.groupby("machine_id").agg(
        failure_count=("failure_id", "count"),
        total_repair_cost=("repair_cost", "sum"),
        total_repair_hours=("downtime_hours", "sum")
    ).reset_index()

    # Aggregate Maintenance
    maint_agg = df_maint.groupby("machine_id").agg(
        maintenance_events=("maintenance_id", "count"),
        total_maintenance_spend=("maintenance_cost", "sum")
    ).reset_index()

    # Merge with KPI frame
    merged = pd.merge(df_kpi, fail_agg, on="machine_id", how="left").fillna(0)
    merged = pd.merge(merged, maint_agg, on="machine_id", how="left").fillna(0)

    # MTBF (Mean Time Between Failures) = Total Operating Hours / Failure Count
    merged["mtbf_hours"] = np.where(
        merged["failure_count"] > 0,
        (merged["operating_hours"] / merged["failure_count"]).round(1),
        merged["operating_hours"].round(1)
    )

    # MTTR (Mean Time To Repair) = Total Repair Hours / Failure Count
    merged["mttr_hours"] = np.where(
        merged["failure_count"] > 0,
        (merged["total_repair_hours"] / merged["failure_count"]).round(2),
        0.0
    )

    # Failure Rate (Lambda) per 1,000 Operating Hours
    merged["failure_rate_per_1000h"] = np.where(
        merged["mtbf_hours"] > 0,
        ((1.0 / merged["mtbf_hours"]) * 1000.0).round(3),
        0.0
    )

    # Financial Cost Breakdown (in INR ₹)
    merged["downtime_revenue_loss_inr"] = (merged["unplanned_downtime_hours"] * PLANT_HOURLY_REVENUE_LOSS).round(2)
    merged["scrap_loss_inr"] = (merged["total_defect_qty"] * SCRAP_UNIT_COST).round(2)
    merged["total_cost_of_unreliability_inr"] = (
        merged["total_repair_cost"] +
        merged["downtime_revenue_loss_inr"] +
        merged["scrap_loss_inr"]
    ).round(2)

    return merged


def compute_monthly_oee_trends(
    df_prod: pd.DataFrame,
    df_down: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes monthly OEE decomposition to identify performance and quality drift over time.
    """
    df_prod = df_prod.copy()
    df_down = df_down.copy()
    df_prod["timestamp"] = pd.to_datetime(df_prod["timestamp"])
    df_down["start_time"] = pd.to_datetime(df_down["start_time"])

    df_prod["month"] = df_prod["timestamp"].dt.to_period("M").astype(str)
    df_down["month"] = df_down["start_time"].dt.to_period("M").astype(str)

    monthly_prod = df_prod.groupby(["machine_id", "month"]).agg(
        planned_qty=("planned_quantity", "sum"),
        actual_qty=("actual_quantity", "sum"),
        good_qty=("good_quantity", "sum")
    ).reset_index()

    monthly_down = df_down.groupby(["machine_id", "month"]).agg(
        downtime_hours=("duration_minutes", lambda x: x.sum() / 60.0)
    ).reset_index()

    monthly_kpi = pd.merge(monthly_prod, monthly_down, on=["machine_id", "month"], how="left").fillna(0)
    
    # Approx 720 hours/month
    month_hours = 720.0
    monthly_kpi["availability_pct"] = np.clip(((month_hours - monthly_kpi["downtime_hours"]) / month_hours) * 100.0, 0, 100)
    monthly_kpi["quality_pct"] = np.clip((monthly_kpi["good_qty"] / monthly_kpi["actual_qty"]) * 100.0, 0, 100)
    monthly_kpi["performance_pct"] = np.clip((monthly_kpi["actual_qty"] / monthly_kpi["planned_qty"]) * 100.0, 0, 100)
    monthly_kpi["oee_pct"] = (
        (monthly_kpi["availability_pct"] / 100.0) *
        (monthly_kpi["performance_pct"] / 100.0) *
        (monthly_kpi["quality_pct"] / 100.0)
    ) * 100.0

    return monthly_kpi.round(2)


def main():
    print("[1/4] Ingesting Cleaned Datasets...")
    df_machines = pd.read_csv(os.path.join(PROCESSED_DIR, "machines.csv"))
    df_prod = pd.read_parquet(os.path.join(PROCESSED_DIR, "production_logs.parquet"))
    df_down = pd.read_csv(os.path.join(PROCESSED_DIR, "downtime_records.csv"))
    df_failures = pd.read_csv(os.path.join(PROCESSED_DIR, "failure_events.csv"))
    df_maint = pd.read_csv(os.path.join(PROCESSED_DIR, "maintenance_logs.csv"))

    print("[2/4] Calculating Fleet-Wide OEE & Asset Utilization...")
    df_oee = compute_fleet_oee(df_machines, df_prod, df_down)

    print("[3/4] Evaluating MTBF, MTTR, and Financial Scrap/Downtime Impact...")
    df_full_kpi = compute_reliability_and_financials(df_oee, df_failures, df_maint)

    print("[4/4] Generating Temporal Monthly OEE Trends...")
    df_monthly = compute_monthly_oee_trends(df_prod, df_down)

    # Persist Results
    df_full_kpi.to_csv(os.path.join(OUTPUT_KPI_DIR, "fleet_kpi_summary.csv"), index=False)
    df_full_kpi.to_parquet(os.path.join(OUTPUT_KPI_DIR, "fleet_kpi_summary.parquet"), index=False)
    df_monthly.to_csv(os.path.join(OUTPUT_KPI_DIR, "monthly_oee_trends.csv"), index=False)

    print("\nManufacturing KPI Execution Complete.")
    print("-----------------------------------------------------------------------------------------")
    print(f"{'Machine':<8} | {'OEE (%)':<8} | {'Avail (%)':<10} | {'Perf (%)':<9} | {'Qual (%)':<9} | {'MTBF (h)':<9} | {'Cost of Unrel (INR)':<18}")
    print("-----------------------------------------------------------------------------------------")
    for _, row in df_full_kpi.iterrows():
        print(f"{row['machine_id']:<8} | {row['oee_pct']:>7.2f}% | {row['availability_pct']:>9.2f}% | {row['performance_pct']:>8.2f}% | {row['quality_pct']:>8.2f}% | {row['mtbf_hours']:>9.1f} | Rs.{row['total_cost_of_unreliability_inr']:>16,.2f}")
    print("-----------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()