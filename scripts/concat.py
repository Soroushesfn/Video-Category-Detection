import pandas as pd

df1 = pd.read_csv('database/OrgData.csv', encoding='ISO-8859-1', on_bad_lines='skip')
df2 = pd.read_csv('database/NewData.csv', encoding='ISO-8859-1', on_bad_lines='skip')    

df2['publish_time'] = pd.to_datetime(df2['publish_time'])
df2['publish_date'] = df2['publish_time'].dt.date
df2['publish_hour'] = df2['publish_time'].dt.hour
df2.drop(['publish_time', 'thumbnail_link'], inplace=True, axis=1)

df2 = df2[df1.columns]
merged_df = pd.concat([df1, df2], ignore_index=True)
merged_df.to_csv('database/Data.csv', index=False)

print(df1.shape, df2.shape, merged_df.shape)