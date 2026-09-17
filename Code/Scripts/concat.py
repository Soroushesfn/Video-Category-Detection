"""Merge the original and supplemental YouTube CSV exports."""

import pandas as pd

from paths import DATABASE_DIR


def main() -> None:
    original = pd.read_csv(
        DATABASE_DIR / "OrgData.csv", encoding="ISO-8859-1", on_bad_lines="skip"
    )
    supplemental = pd.read_csv(
        DATABASE_DIR / "NewData.csv", encoding="ISO-8859-1", on_bad_lines="skip"
    )
    supplemental["publish_time"] = pd.to_datetime(supplemental["publish_time"])
    supplemental["publish_date"] = supplemental["publish_time"].dt.date
    supplemental["publish_hour"] = supplemental["publish_time"].dt.hour
    supplemental = supplemental.drop(columns=["publish_time", "thumbnail_link"])
    merged = pd.concat([original, supplemental[original.columns]], ignore_index=True)
    merged.to_csv(DATABASE_DIR / "Data.csv", index=False)
    print(f"[MERGE] Saved {len(merged):,} rows")


if __name__ == "__main__":
    main()
