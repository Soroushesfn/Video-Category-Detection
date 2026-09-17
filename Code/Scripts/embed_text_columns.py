"""Encode video text fields with a pretrained sentence transformer."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from paths import EMBEDDING_DIR, TMP_DIR, create_runtime_directories


TEXT_COLUMNS = ("title", "tags", "description")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean_tags(text: object) -> str:
    return " ".join(tag.replace('"', "") for tag in str(text).split("|"))


def main() -> None:
    create_runtime_directories()
    dataframe = pd.read_pickle(TMP_DIR / "raw_data.pkl")
    model = SentenceTransformer(MODEL_NAME)

    for column in TEXT_COLUMNS:
        values = dataframe[column].fillna("")
        texts = (
            values.map(clean_tags).tolist()
            if column == "tags"
            else values.astype(str).tolist()
        )
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        output_path = EMBEDDING_DIR / f"{column}_embeddings.npy"
        np.save(output_path, embeddings)
        print(f"[EMBEDDING] Saved {column} embeddings to {output_path}")


if __name__ == "__main__":
    main()
