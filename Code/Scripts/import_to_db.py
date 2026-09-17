"""Import the prepared source CSV into the project SQLite database."""

import sqlite3

import pandas as pd

from paths import DATABASE_DIR


TABLE_NAME = "US_Trending_Videos"


def main() -> None:
    source_path = DATABASE_DIR / "Data.csv"
    output_path = DATABASE_DIR / "dataset.db"
    dataframe = pd.read_csv(source_path, encoding="ISO-8859-1")
    with sqlite3.connect(output_path) as connection:
        dataframe.to_sql(TABLE_NAME, connection, if_exists="replace", index=False)
    print(f"[IMPORT] Saved {len(dataframe):,} records to {output_path}")


if __name__ == "__main__":
    main()
