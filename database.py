import sqlite3
import pandas as pd
import json

# Define the name of the database file
DB_FILE = "maintenance_system.db"

def init_db():
    """
    Initializes the database.
    Creates the 'reports' table if it doesn't already exist.
    """
    # The 'check_same_thread=False' is crucial for Streamlit compatibility
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        # The 'IF NOT EXISTS' clause prevents errors on subsequent runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                machine_id INTEGER NOT NULL,
                health_score REAL NOT NULL,
                failure_prob REAL NOT NULL,
                rul REAL NOT NULL,
                action INTEGER NOT NULL,
                explanation TEXT NOT NULL
            )
        """)
        conn.commit()

def add_report(report: dict):
    """
    Adds a new monitoring report to the database.
    The report dictionary is broken down and inserted into the table.
    """
    # The 'check_same_thread=False' prevents threading errors in Streamlit
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (timestamp, machine_id, health_score, failure_prob, rul, action, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            report['timestamp'],
            report['machine_id'],
            report['health_metrics']['health_score'],
            report['health_metrics']['failure_prob'],
            report['health_metrics']['rul'],
            report['maintenance_action']['action'],
            json.dumps(report['explanation']) # Store the explanation dictionary as a JSON string
        ))
        conn.commit()

def get_reports_by_machine(machine_id: int) -> pd.DataFrame:
    """
    Retrieves all historical reports for a specific machine.
    Returns the data as a pandas DataFrame for easy analysis and plotting.
    """
    # The 'check_same_thread=False' is needed here as well
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        # Use a parameterized query to prevent SQL injection vulnerabilities
        query = "SELECT * FROM reports WHERE machine_id = ? ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn, params=(machine_id,))
        return df
