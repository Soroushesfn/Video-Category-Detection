from sklearn.model_selection import train_test_split
import os
import pandas as pd

folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp", "preprocess.pkl"))

data_df = pd.read_pickle(folder_path)
cat_count = len([col for col in data_df.columns if col.startswith('cat_')])
data_df.columns = list(data_df.columns[: len(data_df.columns) - cat_count]) + ['cat_' + str(num) for num in range(cat_count)]

df_deduped = data_df.sort_values('views', ascending=False).drop_duplicates(subset='video_id', keep='first')

remove_columns = ['video_id', 'channel_title', 'trending_date', 'publish_date', 'publish_hour']
tmp_df = df_deduped.drop(remove_columns, axis=1)
tmp_df = tmp_df[tmp_df['cat_15'] != 1]
y_df = tmp_df[['cat_' + str(num) for num in range(cat_count)]]
X_df = tmp_df.drop(['cat_' + str(num) for num in range(cat_count)], axis=1)



y_class_labels = y_df.values.argmax(axis=1)
(X_trainval, X_test,
  y_trainval, y_test) = train_test_split(
    X_df, y_df,
    test_size=0.2, random_state=44, stratify=y_class_labels)

y_trainval_labels = y_trainval.values.argmax(axis=1)
(X_train, X_val,
 y_train, y_val) = train_test_split(
    X_trainval, y_trainval,
    test_size=0.125, random_state=44, stratify=y_trainval_labels)


tmp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp"))
X_train.to_pickle(tmp_path+"/X_train.pkl")
X_val.to_pickle(tmp_path+"/X_val.pkl")
X_test.to_pickle(tmp_path+"/X_test.pkl")
y_train.to_pickle(tmp_path+"/y_train.pkl")
y_test.to_pickle(tmp_path+"/y_test.pkl")
y_val.to_pickle(tmp_path+"/y_val.pkl")