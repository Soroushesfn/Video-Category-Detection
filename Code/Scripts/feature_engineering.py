"""Combine semantic embeddings with engineered engagement features."""

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from paths import EMBEDDING_DIR, TMP_DIR


def count_tags(tag_string: object) -> int:
    if pd.isna(tag_string):
        return 0
    return sum(bool(tag.strip()) for tag in str(tag_string).split("|"))


def extract_date(date_string: object) -> pd.Timestamp:
    try:
        return pd.Timestamp(
            datetime.strptime(str(date_string).split(",")[1].strip(), "%d %B %Y")
        )
    except (IndexError, TypeError, ValueError):
        return pd.NaT


def main() -> None:
    raw = pd.read_pickle(TMP_DIR / "raw_data.pkl").reset_index(drop=True)
    dataframe = raw.copy()

    for source, prefix in (
        ("title", "title_emb"),
        ("tags", "tags_emb"),
        ("description", "desc_emb"),
    ):
        embeddings = np.load(EMBEDDING_DIR / f"{source}_embeddings.npy")
        columns = [f"{prefix}_{index}" for index in range(embeddings.shape[1])]
        dataframe = pd.concat(
            [dataframe, pd.DataFrame(embeddings, columns=columns)], axis=1
        )

    dataframe = dataframe.drop(columns=["title", "tags", "description"])
    dataframe["engagement_rate"] = (
        dataframe["likes"] + dataframe["dislikes"] + dataframe["comment_count"]
    ) / (dataframe["views"] + 1)
    dataframe["like_dislike_ratio"] = dataframe["likes"] / (
        dataframe["dislikes"] + 1
    )
    dataframe["tag_count"] = raw["tags"].map(count_tags)
    dataframe["publish_date"] = raw["publish_date"].map(extract_date)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(dataframe[["category_id"]])
    category_columns = [f"cat_{int(value)}" for value in encoder.categories_[0]]
    dataframe = pd.concat(
        [
            dataframe.drop(columns=["category_id"]),
            pd.DataFrame(encoded, columns=category_columns),
        ],
        axis=1,
    )

    for column in (
        "comments_disabled",
        "ratings_disabled",
        "video_error_or_removed",
    ):
        dataframe[column] = dataframe[column].astype(int)

    output_path = TMP_DIR / "engineered_features.pkl"
    dataframe.to_pickle(output_path)
    print(f"[FEATURES] Saved {dataframe.shape} feature matrix to {output_path}")


if __name__ == "__main__":
    main()
