import pandas as pd
import sqlite3
import os

# Configuration
CSV_FILE_PATH = "tmp/prediction.csv"
DB_FILE_PATH = "database/prediction.db"
TABLE_NAME = "Prediction"

# Read CSV
print("[IMPORT_TO_DB][ACTION]: Reading CSV file...")
df = pd.read_csv(CSV_FILE_PATH, encoding='ISO-8859-1')

# Create database
print(f"[IMPORT_TO_DB][ACTION]: Creating database at: {DB_FILE_PATH}")
conn = sqlite3.connect(DB_FILE_PATH)
cursor = conn.cursor()

# Drop the table if existing
print(f"[IMPORT_TO_DB][ACTION]: Dropping and creating table '{TABLE_NAME}'...")
cursor.execute(f"DROP TABLE IF EXISTS `{TABLE_NAME}`")

# Create Table based on dataframe
create_cols = []
for col in df.columns:
    dtype = df[col].dtype
    if dtype == "int64":
        sql_type = "INT"
    elif dtype == "float64":
        sql_type = "FLOAT"
    else:
        sql_type = "TEXT"
    create_cols.append(f"`{col}` {sql_type}")
create_table_sql = f"CREATE TABLE `{TABLE_NAME}` ({', '.join(create_cols)})"
cursor.execute(create_table_sql)

# Insert data
print(f"[IMPORT_TO_DB][ACTION]: Inserting data into table '{TABLE_NAME}'...")
df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)

# Cleanup
conn.commit()
conn.close()
print("[IMPORT_TO_DB][ACTION]: Done. Database file created and populated.")
