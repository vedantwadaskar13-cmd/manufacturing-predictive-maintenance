USE manufacturing_intel;

-- =============================================================================
-- Query 1: Fleet-Wide Overall Equipment Effectiveness (OEE) Decomposition
-- Business Question: What is the Availability, Performance, Quality, and OEE per machine?
-- =============================================================================
WITH ProductionSummary AS (
    SELECT 
        machine_id,
        SUM(planned_quantity) AS total_planned_qty,
        SUM(actual_quantity) AS total_actual_qty,
        SUM(good_quantity) AS total_good_qty,
        SUM(defect_quantity) AS total_defect_qty,
        COUNT(DISTINCT DATE(timestamp)) * 24.0 AS total_planned_production_hours
    FROM production_logs
    GROUP BY machine_id
),
DowntimeSummary AS (
    SELECT 
        machine_id,
        SUM(duration_minutes) / 60.0 AS total_downtime_hours
    FROM downtime_records
    GROUP BY machine_id
)
SELECT 
    m.machine_id,
    m.machine_name,
    m.machine_type,
    ROUND(((ps.total_planned_production_hours - COALESCE(dt.total_downtime_hours, 0)) / ps.total_planned_production_hours) * 100, 2) AS availability_pct,
    ROUND((ps.total_actual_qty / ps.total_planned_qty) * 100, 2) AS performance_pct,
    ROUND((ps.total_good_qty / ps.total_actual_qty) * 100, 2) AS quality_pct,
    ROUND(
        (
            ((ps.total_planned_production_hours - COALESCE(dt.total_downtime_hours, 0)) / ps.total_planned_production_hours) *
            (ps.total_actual_qty / ps.total_planned_qty) *
            (ps.total_good_qty / ps.total_actual_qty)
        ) * 100, 
        2
    ) AS oee_pct
FROM machines m
JOIN ProductionSummary ps ON m.machine_id = ps.machine_id
LEFT JOIN DowntimeSummary dt ON m.machine_id = dt.machine_id
ORDER BY oee_pct ASC;

-- =============================================================================
-- Query 2: Reliability Metrics (MTBF & MTTR)
-- Business Question: What is the Mean Time Between Failures and Mean Time to Repair?
-- =============================================================================
WITH ReliabilityData AS (
    SELECT 
        m.machine_id,
        m.machine_name,
        180 * 24.0 AS total_operating_window_hours,
        COALESCE(SUM(f.downtime_hours), 0) AS total_repair_hours,
        COUNT(f.failure_id) AS failure_count
    FROM machines m
    LEFT JOIN failure_events f ON m.machine_id = f.machine_id
    GROUP BY m.machine_id, m.machine_name
)
SELECT 
    machine_id,
    machine_name,
    failure_count,
    ROUND(total_repair_hours, 2) AS total_downtime_hours,
    CASE 
        WHEN failure_count = 0 THEN 4320.0
        ELSE ROUND((total_operating_window_hours - total_repair_hours) / failure_count, 2)
    END AS mtbf_hours,
    CASE 
        WHEN failure_count = 0 THEN 0.0
        ELSE ROUND(total_repair_hours / failure_count, 2)
    END AS mttr_hours
FROM ReliabilityData
ORDER BY failure_count DESC, mtbf_hours ASC;

-- =============================================================================
-- Query 3: Pre-Failure Sensor Behavior (Precursor Drift Analysis)
-- Business Question: How do vibration and temperature compare in the 24h leading up to failure vs baseline?
-- =============================================================================
WITH FailureWindows AS (
    SELECT 
        f.failure_id,
        f.machine_id,
        f.failure_timestamp,
        DATE_SUB(f.failure_timestamp, INTERVAL 24 HOUR) AS window_start
    FROM failure_events f
)
SELECT 
    st.machine_id,
    ROUND(AVG(st.temperature), 2) AS prefailure_avg_temp,
    ROUND(AVG(st.vibration), 3) AS prefailure_avg_vibration,
    ROUND(MAX(st.vibration), 3) AS prefailure_peak_vibration
FROM sensor_telemetry st
JOIN FailureWindows fw ON st.machine_id = fw.machine_id 
    AND st.timestamp BETWEEN fw.window_start AND fw.failure_timestamp
GROUP BY st.machine_id;

-- =============================================================================
-- Query 4: Maintenance Spend & Criticality Ranking
-- Business Question: Which machines consume the highest emergency repair budget?
-- =============================================================================
SELECT 
    m.machine_id,
    m.machine_name,
    m.machine_type,
    COUNT(f.failure_id) AS total_breakdowns,
    COALESCE(SUM(f.repair_cost), 0) AS total_repair_cost,
    COALESCE(SUM(ml.maintenance_cost), 0) AS total_maintenance_spend,
    ROUND(COALESCE(SUM(f.repair_cost), 0) / NULLIF(COUNT(f.failure_id), 0), 2) AS avg_cost_per_failure
FROM machines m
LEFT JOIN failure_events f ON m.machine_id = f.machine_id
LEFT JOIN maintenance_logs ml ON m.machine_id = ml.machine_id
GROUP BY m.machine_id, m.machine_name, m.machine_type
ORDER BY total_repair_cost DESC;