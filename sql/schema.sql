-- =============================================================================
-- Precision Manufacturing Plant - Relational Schema DDL
-- Database: manufacturing_intel
-- =============================================================================

CREATE DATABASE IF NOT EXISTS manufacturing_intel
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE manufacturing_intel;

-- Disable foreign key checks during schema recreation
SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS downtime_records;
DROP TABLE IF EXISTS failure_events;
DROP TABLE IF EXISTS maintenance_logs;
DROP TABLE IF EXISTS production_logs;
DROP TABLE IF EXISTS sensor_telemetry;
DROP TABLE IF EXISTS operators;
DROP TABLE IF EXISTS machines;
SET FOREIGN_KEY_CHECKS = 1;

-- 1. MACHINES (Dimension Table)
CREATE TABLE machines (
    machine_id VARCHAR(10) PRIMARY KEY,
    machine_code VARCHAR(30) NOT NULL UNIQUE,
    machine_name VARCHAR(100) NOT NULL,
    machine_type VARCHAR(50) NOT NULL,
    rated_power_kw DECIMAL(6, 2) NOT NULL,
    installation_date DATE NOT NULL,
    location_bay VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'Active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 2. OPERATORS (Dimension Table)
CREATE TABLE operators (
    operator_id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    shift VARCHAR(20) NOT NULL,
    experience_years INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 3. SENSOR TELEMETRY (Fact Table - High Frequency Time-Series)
CREATE TABLE sensor_telemetry (
    reading_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    machine_id VARCHAR(10) NOT NULL,
    timestamp DATETIME NOT NULL,
    operator_id VARCHAR(10) NOT NULL,
    temperature DECIMAL(5, 2) NOT NULL,
    vibration DECIMAL(6, 3) NOT NULL,
    pressure DECIMAL(6, 2) NOT NULL,
    rpm DECIMAL(6, 1) NOT NULL,
    current DECIMAL(5, 2) NOT NULL,
    degradation_state DECIMAL(5, 4) NOT NULL,
    FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE,
    FOREIGN KEY (operator_id) REFERENCES operators(operator_id),
    INDEX idx_sensor_machine_time (machine_id, timestamp),
    INDEX idx_sensor_timestamp (timestamp)
) ENGINE=InnoDB;

-- 4. PRODUCTION LOGS (Fact Table - Shift Granularity)
CREATE TABLE production_logs (
    production_id VARCHAR(20) PRIMARY KEY,
    machine_id VARCHAR(10) NOT NULL,
    timestamp DATETIME NOT NULL,
    shift VARCHAR(20) NOT NULL,
    operator_id VARCHAR(10) NOT NULL,
    planned_quantity INT UNSIGNED NOT NULL,
    actual_quantity INT UNSIGNED NOT NULL,
    good_quantity INT UNSIGNED NOT NULL,
    defect_quantity INT UNSIGNED NOT NULL,
    actual_cycle_time_sec DECIMAL(6, 2) NOT NULL,
    FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE,
    FOREIGN KEY (operator_id) REFERENCES operators(operator_id),
    INDEX idx_prod_machine_time (machine_id, timestamp),
    INDEX idx_prod_shift (shift)
) ENGINE=InnoDB;

-- 5. MAINTENANCE LOGS (Fact Table - Intervention Events)
CREATE TABLE maintenance_logs (
    maintenance_id VARCHAR(20) PRIMARY KEY,
    machine_id VARCHAR(10) NOT NULL,
    maintenance_date DATETIME NOT NULL,
    maintenance_type ENUM('Preventive', 'Corrective') NOT NULL,
    duration_hours DECIMAL(5, 2) NOT NULL,
    maintenance_cost DECIMAL(10, 2) NOT NULL,
    description TEXT,
    FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE,
    INDEX idx_maint_machine_date (machine_id, maintenance_date)
) ENGINE=InnoDB;

-- 6. FAILURE EVENTS (Fact Table - Ground Truth Breakdown Incidents)
CREATE TABLE failure_events (
    failure_id VARCHAR(20) PRIMARY KEY,
    machine_id VARCHAR(10) NOT NULL,
    failure_timestamp DATETIME NOT NULL,
    failure_type VARCHAR(100) NOT NULL,
    severity ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') NOT NULL,
    downtime_hours DECIMAL(5, 2) NOT NULL,
    repair_cost DECIMAL(10, 2) NOT NULL,
    root_cause VARCHAR(255) NOT NULL,
    FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE,
    INDEX idx_failure_machine_time (machine_id, failure_timestamp)
) ENGINE=InnoDB;

-- 7. DOWNTIME RECORDS (Fact Table - Stoppage Accounting)
CREATE TABLE downtime_records (
    downtime_id VARCHAR(20) PRIMARY KEY,
    machine_id VARCHAR(10) NOT NULL,
    failure_id VARCHAR(20) NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    duration_minutes DECIMAL(8, 2) NOT NULL,
    downtime_type VARCHAR(50) NOT NULL,
    reason VARCHAR(255) NOT NULL,
    planned_flag TINYINT(1) NOT NULL DEFAULT 0,
    FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE,
    FOREIGN KEY (failure_id) REFERENCES failure_events(failure_id) ON DELETE SET NULL,
    INDEX idx_down_machine_time (machine_id, start_time)
) ENGINE=InnoDB;
