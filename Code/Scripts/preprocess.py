"""Normalize numeric features and remove redundant fields."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from paths import TMP_DIR


NUMERIC_COLUMNS = (
    "views",
    "publish_hour",
    "likes",
    "dislikes",
    "comment_count",
    "engagement_rate",
    "like_dislike_ratio",
    "tag_count",
)


def main() -> None:
    dataframe = pd.read_pickle(TMP_DIR / "engineered_features.pkl")
    dataframe = dataframe.drop(
        columns=[
            "comments_disabled",
            "ratings_disabled",
            "video_error_or_removed",
        ]
    ).drop_duplicates()

    numeric = dataframe.loc[:, NUMERIC_COLUMNS].apply(np.log1p)
    numeric = MinMaxScaler().fit_transform(numeric)
    dataframe.loc[:, NUMERIC_COLUMNS] = StandardScaler().fit_transform(numeric)
    dataframe = dataframe.drop(columns=["likes", "dislikes"])

    output_path = TMP_DIR / "preprocessed.pkl"
    dataframe.to_pickle(output_path)
    print(f"[PREPROCESS] Saved {dataframe.shape} matrix to {output_path}")


if __name__ == "__main__":
    main()
