import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv('database/Data.csv', encoding='ISO-8859-1')
print(df['categoryID'].unique())