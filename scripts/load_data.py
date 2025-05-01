
import sqlite3
import pandas as pd
import os

db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "dataset.db"))
conn = sqlite3.connect(db_path)


query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql(query, conn)

dataframes = {}
for table_name in tables['name']:
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    dataframes[table_name] = df


conn.close()
