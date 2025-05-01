import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv('database/Data.csv', encoding='ISO-8859-1')
sample_channel = df['channel_title'][0]
print(sample_channel)

model = SentenceTransformer('all-MiniLM-L6-v2')
title_embeddings = df['title'].fillna('').apply(model.encode)
print(title_embeddings)