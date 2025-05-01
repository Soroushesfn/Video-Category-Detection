import pandas as pd

df = pd.read_pickle('tmp/raw_data.pkl')
print(df['category_id'].unique())