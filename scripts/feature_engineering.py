import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import os

# Calculates the count of used tages
def count_tags(tag_str):
    if pd.isna(tag_str):
        return 0
    return len([t for t in tag_str.split('|') if t.strip() != ""])

# Extracts date
def extract_date(date_str):
    try:
        return datetime.strptime(date_str.split(",")[1].strip(), "%d %B %Y")
    except Exception:
        return pd.NaT

# Paths
RAW_DATA_PATH = 'tmp/raw_data.pkl'
EMB_DIR = 'tmp/embeddings/'
OUTPUT_PATH = 'tmp/engineered_features.pkl'

# Load data
print("[FEATURE_ENGINEERING][INFO]: Loading raw data...")
df = pd.read_pickle(RAW_DATA_PATH)

# Load embeddings
print("[FEATURE_ENGINEERING][INFO]: Loading precomputed embeddings...")
title_emb = np.load(os.path.join(EMB_DIR, 'title_embeddings.npy'))
tags_emb = np.load(os.path.join(EMB_DIR, 'tags_embeddings.npy'))
desc_emb = np.load(os.path.join(EMB_DIR, 'description_embeddings.npy'))

# Replace text with embeddings
print("[FEATURE_ENGINEERING][INFO]: Replacing text columns with embeddings...")
title_df = pd.DataFrame(title_emb, columns=[f'title_emb_{i}' for i in range(title_emb.shape[1])])
tags_df = pd.DataFrame(tags_emb, columns=[f'tags_emb_{i}' for i in range(tags_emb.shape[1])])
desc_df = pd.DataFrame(desc_emb, columns=[f'desc_emb_{i}' for i in range(desc_emb.shape[1])])
df.drop(columns=["title", "tags", "description"], inplace=True)
df = pd.concat([df.reset_index(drop=True), title_df, tags_df, desc_df], axis=1)

# Create new columns
print("[FEATURE_ENGINEERING][INFO]: Creating new feature columns...")
# Engagement rate
df["engagement_rate"] = (df["likes"] + df["dislikes"] + df["comment_count"]) / (df["views"] + 1)
# Like-dislike ratio
df["like_dislike_ratio"] = df["likes"] / (df["dislikes"] + 1)
# Count of tags
raw_df = pd.read_pickle(RAW_DATA_PATH)
df["tag_count"] = raw_df["tags"].apply(count_tags)

# Encode category id
print("[FEATURE_ENGINEERING][INFO]: Encoding category_id...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
category_encoded = encoder.fit_transform(df[["category_id"]])
category_df = pd.DataFrame(category_encoded, columns=[f"cat_{int(cat)}" for cat in encoder.categories_[0]])
df.drop(columns=["category_id"], inplace=True)
df = pd.concat([df.reset_index(drop=True), category_df], axis=1)

# Correct publish date
print("[FEATURE_ENGINEERING][INFO]: Parsing publish_date...")
df["publish_date"] = raw_df["publish_date"].apply(extract_date)

# Convert booleans to binaries
print("[FEATURE_ENGINEERING][INFO]: Encoding boolean columns...")
bool_cols = ["comments_disabled", "ratings_disabled", "video_error_or_removed"]
for col in bool_cols:
    df[col] = df[col].astype(int)

# # Save result
print(f"[SUCCESS] Saving final features to: {OUTPUT_PATH}")
df.to_pickle(OUTPUT_PATH)