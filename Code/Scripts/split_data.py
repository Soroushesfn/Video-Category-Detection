"""Create stratified train, validation, and test partitions."""

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import TMP_DIR


def main() -> None:
    dataframe = pd.read_pickle(TMP_DIR / "preprocessed.pkl")
    dataframe = (
        dataframe.sort_values("views", ascending=False)
        .drop_duplicates(subset="video_id", keep="first")
        .reset_index(drop=True)
    )

    target_columns = [column for column in dataframe if column.startswith("cat_")]
    excluded = [
        "video_id",
        "channel_title",
        "trending_date",
        "publish_date",
        "publish_hour",
    ]
    features = dataframe.drop(columns=excluded + target_columns)
    targets = dataframe[target_columns]

    # Category 15 was not represented sufficiently for a stratified split.
    if "cat_15" in targets:
        keep = targets["cat_15"].ne(1)
        features, targets = features.loc[keep], targets.loc[keep]

    labels = targets.to_numpy().argmax(axis=1)
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=44,
        stratify=labels,
    )
    trainval_labels = y_trainval.to_numpy().argmax(axis=1)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=0.125,
        random_state=44,
        stratify=trainval_labels,
    )

    for name, split in (
        ("X_train", x_train),
        ("X_val", x_val),
        ("X_test", x_test),
        ("y_train", y_train),
        ("y_val", y_val),
        ("y_test", y_test),
    ):
        split.to_pickle(TMP_DIR / f"{name}.pkl")

    print(
        "[SPLIT] Saved train/validation/test sets: "
        f"{len(x_train):,}/{len(x_val):,}/{len(x_test):,}"
    )


if __name__ == "__main__":
    main()
