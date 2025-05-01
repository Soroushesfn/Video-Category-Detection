import sqlite3
import os

def get_db_connection(db_path):
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to connect to database: {e}")
        return None

