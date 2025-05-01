import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


numeric_attributes = list(["views", "publish_hour", "likes", "dislikes", "comment_count", "engagement_rate", "like_dislike_ratio", "tag_count"])

eng_feature_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp", "engineered_features.pkl"))
df = pd.read_pickle(eng_feature_path)

# unvaluable columns
df = df.drop(['comments_disabled','ratings_disabled', 'video_error_or_removed'], axis=1)

df = df.drop_duplicates()

# normalization and standardization
df[numeric_attributes] = df[numeric_attributes].apply(lambda x: np.log1p(x))

scaler = MinMaxScaler()
df[numeric_attributes] = scaler.fit_transform(df[numeric_attributes])

scaler = StandardScaler()
df[numeric_attributes] = scaler.fit_transform(df[numeric_attributes] )

# correlation
df = df.drop(["likes", "dislikes"], axis=1)

preprocess_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp", "preprocess.pkl"))
df.to_pickle(preprocess_path)
