"""
Precision Manufacturing Plant - MySQL Database Loader Pipeline
Executes schema initialization, loads cleaned CSVs in dependency order,
and creates analytical database views.
"""

import os
import sys
import time
import pandas as pd
from sqlalchemy import create_engine, text

# Configuration - Override via environment variables if needed
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "Vw,130505")
DB_HOST = os.getenv("MYSQL_HOST", "localhost")
DB_PORT = os.getenv("MYSQL_PORT", "3306")
DB_NAME = os.getenv("MYSQL_DATABASE", "manufacturing_intel")

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
SQL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sql")

def get_db_engine(include_db=True):
    db_suffix = f"/{DB_NAME}" if include_db else ""
    conn_str = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}{db_suffix}?charset=utf8mb4"
    return create_engine(conn_str, echo=False)

def execute_sql_file(engine, file_path):
    print(f"Executing SQL script: {os.path.basename(file_path)}...")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_sql = f.read()

    # Split into discrete statements
    statements = [stmt.strip() for stmt in raw_sql.split(";") if stmt.strip()]
    with engine.connect() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
        conn.commit()
    print(f"Successfully applied {os.path.basename(file_path)}")

def load_table_data(engine, table_name, csv_filename, chunksize=5000):
    csv_path = os.path.join(PROCESSED_DIR, csv_filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing processed file: {csv_path}")

    print(f"Loading table '{table_name}' from {csv_filename}...")
    df = pd.read_csv(csv_path)
    
    # Write to MySQL table
    start_t = time.time()
    df.to_sql(name=table_name, con=engine, if_exists="append", index=False, chunksize=chunksize)
    elapsed = time.time() - start_t
    print(f"  -> Inserted {len(df):>6,d} rows into '{table_name}' in {elapsed:.2f}s")

def verify_database_counts(engine):
    tables = [
        "machines", "operators", "sensor_telemetry", "production_logs",
        "maintenance_logs", "failure_events", "downtime_records"
    ]
    print("\nDatabase Row Count Verification:")
    print("--------------------------------------------------")
    with engine.connect() as conn:
        for tbl in tables:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {tbl};")).scalar()
            print(f"  Table: {tbl:<20} | Rows: {result:>8,d}")
    print("--------------------------------------------------")

def main():
    try:
        # 1. Connect to MySQL server root to create database
        root_engine = get_db_engine(include_db=False)
        with root_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};"))
            conn.commit()

        # 2. Connect to target database and apply DDL schema
        engine = get_db_engine(include_db=True)
        schema_file = os.path.join(SQL_DIR, "schema.sql")
        execute_sql_file(engine, schema_file)

        # 3. Load tables in relational dependency order
        load_table_data(engine, "machines", "machines.csv")
        load_table_data(engine, "operators", "operators.csv")
        load_table_data(engine, "sensor_telemetry", "sensor_telemetry.csv", chunksize=10000)
        load_table_data(engine, "production_logs", "production_logs.csv")
        load_table_data(engine, "maintenance_logs", "maintenance_logs.csv")
        load_table_data(engine, "failure_events", "failure_events.csv")
        load_table_data(engine, "downtime_records", "downtime_records.csv")

        # 4. Create Materialized Analytical Views
        views_file = os.path.join(SQL_DIR, "views.sql")
        execute_sql_file(engine, views_file)

        # 5. Verify record counts
        verify_database_counts(engine)
        print("Phase 3 Database Setup & Ingestion Complete.")

    except Exception as e:
        print(f"\n[ERROR] Database loading failed: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()