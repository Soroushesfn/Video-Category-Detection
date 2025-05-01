import sqlite3
import pandas as pd
import os
from database_connection import get_db_connection

TABLE_NAME = 'US_Trending_Videos'
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "dataset.db"))
conn = get_db_connection(db_path)

if conn:
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
else:
    print("Database connection failed.")

dataframes = {}
for table_name in tables['name']:
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    dataframes[table_name] = df

# Save as pickle
dataframes['US_Trending_Videos'].to_pickle('tmp/raw_data.pkl')

if conn:
    conn.close()