"""
Precision Manufacturing Plant - Data Generation Engine
Generates multi-table relational manufacturing data with realistic physics,
degradation trajectories, maintenance actions, and failure mechanisms.
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Set reproducible random seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "synthetic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration Parameters
START_DATE = datetime(2025, 1, 1, 0, 0, 0)
DURATION_DAYS = 180  # 6 Months of operational data
TIMESTAMPS = [START_DATE + timedelta(hours=i) for i in range(DURATION_DAYS * 24)]

FLEET_CONFIG = {
    "M01": {"name": "CNC Lathe 01", "type": "CNC Lathe", "power_kw": 18.5, "base_temp": 42.0, "base_vib": 1.1, "base_press": 0.0, "base_rpm": 2200, "base_curr": 14.0, "mttf_days": 42},
    "M02": {"name": "CNC Milling 01", "type": "CNC Milling", "power_kw": 22.0, "base_temp": 45.0, "base_vib": 1.3, "base_press": 0.0, "base_rpm": 3000, "base_curr": 18.5, "mttf_days": 38},
    "M03": {"name": "CNC Milling 02", "type": "CNC Milling", "power_kw": 30.0, "base_temp": 48.0, "base_vib": 1.4, "base_press": 0.0, "base_rpm": 2800, "base_curr": 22.0, "mttf_days": 45},
    "M04": {"name": "Hydraulic Press 01", "type": "Hydraulic Press", "power_kw": 45.0, "base_temp": 52.0, "base_vib": 2.2, "base_press": 140.0, "base_rpm": 0, "base_curr": 35.0, "mttf_days": 30},
    "M05": {"name": "Hydraulic Unit 02", "type": "Hydraulic System", "power_kw": 37.0, "base_temp": 50.0, "base_vib": 1.9, "base_press": 120.0, "base_rpm": 0, "base_curr": 28.0, "mttf_days": 35},
    "M06": {"name": "Coolant Pump 01", "type": "Pump", "power_kw": 7.5, "base_temp": 38.0, "base_vib": 0.8, "base_press": 6.5, "base_rpm": 1450, "base_curr": 8.2, "mttf_days": 55},
    "M07": {"name": "Slurry Pump 02", "type": "Pump", "power_kw": 11.0, "base_temp": 40.0, "base_vib": 1.0, "base_press": 8.0, "base_rpm": 1450, "base_curr": 11.5, "mttf_days": 40},
    "M08": {"name": "Drive Motor 01", "type": "Electric Motor", "power_kw": 55.0, "base_temp": 55.0, "base_vib": 1.2, "base_press": 0.0, "base_rpm": 1780, "base_curr": 45.0, "mttf_days": 60},
    "M09": {"name": "Drive Motor 02", "type": "Electric Motor", "power_kw": 75.0, "base_temp": 58.0, "base_vib": 1.5, "base_press": 0.0, "base_rpm": 1780, "base_curr": 62.0, "mttf_days": 50},
    "M10": {"name": "Screw Compressor 01", "type": "Compressor", "power_kw": 90.0, "base_temp": 68.0, "base_vib": 2.5, "base_press": 8.5, "base_rpm": 3600, "base_curr": 78.0, "mttf_days": 28},
}

OPERATORS_DATA = [
    {"operator_id": "OP01", "name": "Rajesh Sharma", "shift": "Shift 1", "experience_years": 8},
    {"operator_id": "OP02", "name": "Amit Patil", "shift": "Shift 2", "experience_years": 5},
    {"operator_id": "OP03", "name": "Suresh Deshmukh", "shift": "Shift 3", "experience_years": 12},
    {"operator_id": "OP04", "name": "Pooja Kulkarni", "shift": "Shift 1", "experience_years": 3},
    {"operator_id": "OP05", "name": "Vikas Jadhav", "shift": "Shift 2", "experience_years": 6},
    {"operator_id": "OP06", "name": "Sunil Shinde", "shift": "Shift 3", "experience_years": 2},
]


def generate_static_tables():
    # 1. Machines Table
    machines_list = []
    for m_id, cfg in FLEET_CONFIG.items():
        machines_list.append({
            "machine_id": m_id,
            "machine_code": f"PREC-{m_id}",
            "machine_name": cfg["name"],
            "machine_type": cfg["type"],
            "rated_power_kw": cfg["power_kw"],
            "installation_date": "2022-03-15",
            "location_bay": f"Bay-0{int(m_id[1:]) % 3 + 1}",
            "status": "Active"
        })
    df_machines = pd.DataFrame(machines_list)

    # 2. Operators Table
    df_operators = pd.DataFrame(OPERATORS_DATA)

    return df_machines, df_operators


def generate_plant_simulation():
    sensor_rows = []
    production_rows = []
    maintenance_rows = []
    failure_rows = []
    downtime_rows = []

    reading_id = 1
    prod_id = 1
    maint_id = 1
    fail_id = 1
    down_id = 1

    for m_id, cfg in FLEET_CONFIG.items():
        current_time = START_DATE
        hours_since_last_maint = random.randint(20, 150)
        mttf_hours = cfg["mttf_days"] * 24
        
        # Track continuous health state (0.0 = Brand New, 1.0 = Failure Threshold)
        degradation_level = hours_since_last_maint / (mttf_hours * 1.3)

        while current_time < (START_DATE + timedelta(days=DURATION_DAYS)):
            hour_of_day = current_time.hour
            
            # Determine Shift and Operator
            if 6 <= hour_of_day < 14:
                shift = "Shift 1"
                op_id = "OP01" if int(m_id[1:]) % 2 == 1 else "OP04"
            elif 14 <= hour_of_day < 22:
                shift = "Shift 2"
                op_id = "OP02" if int(m_id[1:]) % 2 == 1 else "OP05"
            else:
                shift = "Shift 3"
                op_id = "OP03" if int(m_id[1:]) % 2 == 1 else "OP06"

            # Diurnal thermal wave
            ambient_thermal_drift = 3.5 * np.sin((2 * np.pi * hour_of_day / 24) - (np.pi / 2))
            
            # Operational degradation advancement per hour
            wear_rate = (1.0 / mttf_hours) * np.random.uniform(0.7, 1.3)
            degradation_level += wear_rate
            hours_since_last_maint += 1

            # Failure condition evaluation
            is_breakdown = False
            if degradation_level >= 1.0 or (degradation_level > 0.75 and random.random() < 0.08):
                is_breakdown = True

            # Synthesis of physical parameters
            # Exponential wear curve: small impact early, accelerates rapidly near failure
            wear_multiplier = np.exp(2.8 * degradation_level) - 1.0

            # 1. Temperature: Base + Ambient + Friction/Load Heat + Noise
            temp_noise = np.random.normal(0, 0.6)
            temp = cfg["base_temp"] + ambient_thermal_drift + (wear_multiplier * 8.5) + temp_noise

            # 2. Vibration (RMS mm/s): Primary degradation indicator
            vib_noise = np.random.normal(0, 0.08)
            vib = cfg["base_vib"] + (wear_multiplier * 2.8) + vib_noise

            # 3. Pressure (bar)
            if cfg["base_press"] > 0:
                # Hydraulic/Compressor pressure degrades downward or fluctuates wildly
                press_noise = np.random.normal(0, 0.4)
                press = cfg["base_press"] - (wear_multiplier * 4.2) + press_noise
            else:
                press = 0.0

            # 4. RPM
            if cfg["base_rpm"] > 0:
                rpm_slip = wear_multiplier * 25.0
                rpm = cfg["base_rpm"] - rpm_slip + np.random.normal(0, 5.0)
            else:
                rpm = 0.0

            # 5. Current (Amperes): Increases with internal mechanical resistance
            curr_noise = np.random.normal(0, 0.5)
            curr = cfg["base_curr"] + (wear_multiplier * 3.5) + curr_noise

            # Inject sporadic, non-fatal telemetry glitches (0.4% chance)
            if random.random() < 0.004 and not is_breakdown:
                if random.random() < 0.5:
                    temp += np.random.choice([15.0, -10.0])  # Sensor glitch
                else:
                    vib += np.random.choice([3.0, 4.5])     # Temporary mechanical shock

            # Append Telemetry Record
            sensor_rows.append({
                "reading_id": reading_id,
                "machine_id": m_id,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "operator_id": op_id,
                "temperature": round(float(temp), 2),
                "vibration": round(max(0.1, float(vib)), 3),
                "pressure": round(max(0.0, float(press)), 2),
                "rpm": round(max(0.0, float(rpm)), 1),
                "current": round(max(0.0, float(curr)), 2),
                "degradation_state": round(min(1.0, float(degradation_level)), 4)
            })
            reading_id += 1

            # Handle Functional Breakdown Event
            if is_breakdown:
                failure_modes = {
                    "CNC Lathe": ("Spindle Bearing Seizure", "Mechanical Wear", 35000),
                    "CNC Milling": ("Tool Holder Fatigue", "High Dynamic Load", 42000),
                    "Hydraulic Press": ("Main Cylinder Seal Failure", "Hydraulic Overpressure", 58000),
                    "Hydraulic System": ("Proportional Valve Jam", "Fluid Contamination", 29000),
                    "Pump": ("Mechanical Seal Rupture", "Impeller Cavitation", 18000),
                    "Electric Motor": ("Stator Winding Breakdown", "Thermal Degradation", 48000),
                    "Compressor": ("Screw Element Wear", "Lubrication Breakdown", 65000)
                }
                fail_mode, root_cause, base_cost = failure_modes[cfg["type"]]
                repair_hours = random.randint(6, 18)
                repair_cost = base_cost + random.randint(2000, 10000)

                # Record Failure Event
                failure_rows.append({
                    "failure_id": f"FAIL_{fail_id:04d}",
                    "machine_id": m_id,
                    "failure_timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "failure_type": fail_mode,
                    "severity": "CRITICAL" if repair_hours > 12 else "HIGH",
                    "downtime_hours": repair_hours,
                    "repair_cost": repair_cost,
                    "root_cause": root_cause
                })

                # Record Unplanned Downtime
                downtime_rows.append({
                    "downtime_id": f"DT_{down_id:04d}",
                    "machine_id": m_id,
                    "failure_id": f"FAIL_{fail_id:04d}",
                    "start_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": (current_time + timedelta(hours=repair_hours)).strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_minutes": repair_hours * 60,
                    "downtime_type": "Unplanned Breakdown",
                    "reason": f"Catastrophic failure: {fail_mode}",
                    "planned_flag": 0
                })
                down_id += 1

                # Corrective Maintenance Log
                maintenance_rows.append({
                    "maintenance_id": f"MAINT_{maint_id:04d}",
                    "machine_id": m_id,
                    "maintenance_date": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "maintenance_type": "Corrective",
                    "duration_hours": repair_hours,
                    "maintenance_cost": repair_cost,
                    "description": f"Emergency replacement and restoration due to {fail_mode}"
                })
                maint_id += 1
                fail_id += 1

                # Fast-forward operational clock across repair downtime
                current_time += timedelta(hours=repair_hours)
                # Reset health state after maintenance
                degradation_level = np.random.uniform(0.02, 0.08)
                hours_since_last_maint = 0
                continue

            # Handle Planned Preventive Maintenance
            elif hours_since_last_maint > (mttf_hours * 0.85) and random.random() < 0.25:
                pm_duration = random.randint(2, 5)
                pm_cost = random.randint(4000, 12000)

                maintenance_rows.append({
                    "maintenance_id": f"MAINT_{maint_id:04d}",
                    "machine_id": m_id,
                    "maintenance_date": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "maintenance_type": "Preventive",
                    "duration_hours": pm_duration,
                    "maintenance_cost": pm_cost,
                    "description": "Scheduled inspection, fluid replenishment, dynamic balancing"
                })
                maint_id += 1

                downtime_rows.append({
                    "downtime_id": f"DT_{down_id:04d}",
                    "machine_id": m_id,
                    "failure_id": None,
                    "start_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": (current_time + timedelta(hours=pm_duration)).strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_minutes": pm_duration * 60,
                    "downtime_type": "Planned PM",
                    "reason": "Scheduled Preventive Maintenance",
                    "planned_flag": 1
                })
                down_id += 1

                current_time += timedelta(hours=pm_duration)
                degradation_level = np.random.uniform(0.01, 0.05)
                hours_since_last_maint = 0
                continue

            # Production Log Aggregation (Every 8-Hour Shift Mark)
            if hour_of_day in [6, 14, 22]:
                nominal_cycle_sec = 45.0 if "CNC" in cfg["type"] else 30.0
                planned_qty = int((8 * 3600) / nominal_cycle_sec)
                
                # Degraded machines suffer cycle elongation and scrap spikes
                efficiency_factor = max(0.65, 1.0 - (degradation_level * 0.28))
                actual_qty = int(planned_qty * efficiency_factor * np.random.uniform(0.95, 1.0))
                
                scrap_rate = 0.01 + (degradation_level ** 2) * 0.12 + np.random.uniform(0.0, 0.02)
                defect_qty = int(actual_qty * scrap_rate)
                good_qty = actual_qty - defect_qty

                production_rows.append({
                    "production_id": f"PRD_{prod_id:06d}",
                    "machine_id": m_id,
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "shift": shift,
                    "operator_id": op_id,
                    "planned_quantity": planned_qty,
                    "actual_quantity": actual_qty,
                    "good_quantity": good_qty,
                    "defect_quantity": defect_qty,
                    "actual_cycle_time_sec": round(nominal_cycle_sec / max(0.1, efficiency_factor), 2)
                })
                prod_id += 1

            current_time += timedelta(hours=1)

    return (
        pd.DataFrame(sensor_rows),
        pd.DataFrame(production_rows),
        pd.DataFrame(maintenance_rows),
        pd.DataFrame(failure_rows),
        pd.DataFrame(downtime_rows)
    )


def main():
    print("[1/3] Generating Static Fleet and Operator Metadata...")
    df_machines, df_operators = generate_static_tables()

    print("[2/3] Simulating Continuous Industrial Physics, Events, and Production...")
    df_sensor, df_production, df_maintenance, df_failures, df_downtime = generate_plant_simulation()

    print("[3/3] Serializing Synthetic Relational Datasets to CSV...")
    df_machines.to_csv(os.path.join(OUTPUT_DIR, "machines.csv"), index=False)
    df_operators.to_csv(os.path.join(OUTPUT_DIR, "operators.csv"), index=False)
    df_sensor.to_csv(os.path.join(OUTPUT_DIR, "sensor_telemetry.csv"), index=False)
    df_production.to_csv(os.path.join(OUTPUT_DIR, "production_logs.csv"), index=False)
    df_maintenance.to_csv(os.path.join(OUTPUT_DIR, "maintenance_logs.csv"), index=False)
    df_failures.to_csv(os.path.join(OUTPUT_DIR, "failure_events.csv"), index=False)
    df_downtime.to_csv(os.path.join(OUTPUT_DIR, "downtime_records.csv"), index=False)

    print("\nDataset Generation Complete.")
    print("--------------------------------------------------")
    print(f"Machines:           {len(df_machines):>8,d} rows")
    print(f"Operators:          {len(df_operators):>8,d} rows")
    print(f"Sensor Readings:    {len(df_sensor):>8,d} rows")
    print(f"Production Logs:    {len(df_production):>8,d} rows")
    print(f"Maintenance Logs:   {len(df_maintenance):>8,d} rows")
    print(f"Failure Events:     {len(df_failures):>8,d} rows")
    print(f"Downtime Records:   {len(df_downtime):>8,d} rows")
    print(f"Output Location:    {OUTPUT_DIR}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()