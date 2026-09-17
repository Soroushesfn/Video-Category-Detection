"""Load and deduplicate the source dataset from SQLite."""

import pandas as pd

from database_connection import get_db_connection
from paths import DATABASE_DIR, TMP_DIR, create_runtime_directories


TABLE_NAME = "US_Trending_Videos"
DATABASE_PATH = DATABASE_DIR / "dataset.db"


def main() -> None:
    create_runtime_directories()
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Source database not found: {DATABASE_PATH}. "
            "See the README for data setup instructions."
        )

    with get_db_connection(DATABASE_PATH) as connection:
        available = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if TABLE_NAME not in available:
            raise ValueError(f"Table {TABLE_NAME!r} is missing from {DATABASE_PATH}")
        dataframe = pd.read_sql_query(f'SELECT * FROM "{TABLE_NAME}"', connection)

    deduplicated = (
        dataframe.sort_values("views", ascending=False)
        .drop_duplicates(subset="video_id", keep="first")
        .reset_index(drop=True)
    )
    deduplicated.to_pickle(TMP_DIR / "raw_data.pkl")
    print(f"[LOAD] Saved {len(deduplicated):,} unique videos")


if __name__ == "__main__":
    main()
