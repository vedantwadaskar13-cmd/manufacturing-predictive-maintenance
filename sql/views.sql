USE manufacturing_intel;

-- -----------------------------------------------------------------------------
-- VIEW 1: Daily Machine Telemetry & Degradation Summary
-- Provides daily aggregated telemetry baselines to monitor wear drift
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW view_daily_machine_telemetry AS
SELECT 
    m.machine_id,
    m.machine_name,
    m.machine_type,
    DATE(st.timestamp) AS reading_date,
    ROUND(AVG(st.temperature), 2) AS avg_temperature,
    ROUND(MAX(st.temperature), 2) AS max_temperature,
    ROUND(AVG(st.vibration), 3) AS avg_vibration,
    ROUND(MAX(st.vibration), 3) AS max_vibration,
    ROUND(AVG(st.pressure), 2) AS avg_pressure,
    ROUND(AVG(st.current), 2) AS avg_current,
    ROUND(MAX(st.degradation_state), 4) AS max_degradation_state,
    COUNT(st.reading_id) AS total_readings
FROM machines m
JOIN sensor_telemetry st ON m.machine_id = st.machine_id
GROUP BY m.machine_id, m.machine_name, m.machine_type, DATE(st.timestamp);

-- -----------------------------------------------------------------------------
-- VIEW 2: Shift-Level Production & Quality Metrics
-- Pre-calculates scrap rate, defect volume, and production yield per shift
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW view_shift_production_quality AS
SELECT 
    p.production_id,
    p.machine_id,
    m.machine_name,
    p.timestamp,
    DATE(p.timestamp) AS production_date,
    p.shift,
    p.operator_id,
    o.name AS operator_name,
    p.planned_quantity,
    p.actual_quantity,
    p.good_quantity,
    p.defect_quantity,
    ROUND((p.defect_quantity / NULLIF(p.actual_quantity, 0)) * 100, 2) AS defect_rate_pct,
    ROUND((p.actual_quantity / NULLIF(p.planned_quantity, 0)) * 100, 2) AS output_efficiency_pct,
    p.actual_cycle_time_sec
FROM production_logs p
JOIN machines m ON p.machine_id = m.machine_id
JOIN operators o ON p.operator_id = o.operator_id;

-- -----------------------------------------------------------------------------
-- VIEW 3: Machine Downtime & Failure Impact Summary
-- Aggregates planned vs unplanned stoppage duration and financial repair impact
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW view_machine_downtime_summary AS
SELECT 
    m.machine_id,
    m.machine_name,
    m.machine_type,
    COUNT(DISTINCT dt.downtime_id) AS total_downtime_incidents,
    ROUND(SUM(CASE WHEN dt.planned_flag = 1 THEN dt.duration_minutes ELSE 0 END) / 60.0, 2) AS planned_downtime_hours,
    ROUND(SUM(CASE WHEN dt.planned_flag = 0 THEN dt.duration_minutes ELSE 0 END) / 60.0, 2) AS unplanned_downtime_hours,
    ROUND(SUM(dt.duration_minutes) / 60.0, 2) AS total_downtime_hours,
    COUNT(DISTINCT f.failure_id) AS total_failures,
    COALESCE(SUM(f.repair_cost), 0) AS total_failure_repair_cost,
    COALESCE(SUM(ml.maintenance_cost), 0) AS total_maintenance_spend
FROM machines m
LEFT JOIN downtime_records dt ON m.machine_id = dt.machine_id
LEFT JOIN failure_events f ON m.machine_id = f.machine_id
LEFT JOIN maintenance_logs ml ON m.machine_id = ml.machine_id
GROUP BY m.machine_id, m.machine_name, m.machine_type;

-- -----------------------------------------------------------------------------
-- VIEW 4: Latest Machine Operational Health State
-- Fetches the single most recent sensor observation and maintenance timestamp
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW view_latest_machine_status AS
WITH RankedTelemetry AS (
    SELECT 
        reading_id, machine_id, timestamp, temperature, vibration, 
        pressure, rpm, current, degradation_state,
        ROW_NUMBER() OVER (PARTITION BY machine_id ORDER BY timestamp DESC) as rn
    FROM sensor_telemetry
),
LastMaintenance AS (
    SELECT 
        machine_id,
        MAX(maintenance_date) AS last_maintenance_date
    FROM maintenance_logs
    GROUP BY machine_id
)
SELECT 
    m.machine_id,
    m.machine_name,
    m.machine_type,
    m.status AS operational_status,
    rt.timestamp AS last_telemetry_time,
    rt.temperature,
    rt.vibration,
    rt.pressure,
    rt.rpm,
    rt.current,
    rt.degradation_state,
    lm.last_maintenance_date,
    TIMESTAMPDIFF(HOUR, lm.last_maintenance_date, rt.timestamp) AS hours_since_maintenance
FROM machines m
JOIN RankedTelemetry rt ON m.machine_id = rt.machine_id AND rt.rn = 1
LEFT JOIN LastMaintenance lm ON m.machine_id = lm.machine_id;