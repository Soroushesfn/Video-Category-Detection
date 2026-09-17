"""Export model predictions to SQLite for downstream analysis."""

import sqlite3

import pandas as pd

from paths import DATABASE_DIR, TMP_DIR


TABLE_NAME = "Prediction"


def main() -> None:
    dataframe = pd.read_csv(TMP_DIR / "predictions.csv")
    output_path = DATABASE_DIR / "prediction.db"
    with sqlite3.connect(output_path) as connection:
        dataframe.to_sql(TABLE_NAME, connection, if_exists="replace", index=False)
    print(f"[EXPORT] Saved {len(dataframe):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
