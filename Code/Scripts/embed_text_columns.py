import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import pickle

# Meta data
# Columns
TEXTUAL_COLUMNS = ["title", "tags", "description"]
# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_DIR = "tmp/embeddings/"

# Load Data
print("[EMBEDDING][INFO]: Loading data...")
df = pd.read_pickle('tmp/raw_data.pkl') 
print(df.shape)

# Load embedding model
print(f"[EMBEDDING][INFO]: Loading embedding model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

# Clean tags
def clean_tags(text):
    return " ".join(tag.replace('"', '') for tag in str(text).split('|'))

# Embedding each column
for column in TEXTUAL_COLUMNS:
    print(f"[EMBEDDING][INFO]: Embedding column: {column}...")
    
    if column == "tags":
        texts = df[column].fillna("").apply(clean_tags).tolist()
    else:
        texts = df[column].fillna("").astype(str).tolist()
    
    # perform embedding
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # Save
    out_path = OUTPUT_DIR + f"{column}_embeddings.npy"
    np.save(out_path, embeddings)
    print(f"[EMBEDDING][SUCCESS]: Saved embeddings to: {out_path}")