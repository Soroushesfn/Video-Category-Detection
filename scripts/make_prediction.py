import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report


tmp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp"))
X_test = pd.read_pickle(tmp_path+"/X_test.pkl")
y_test = pd.read_pickle(tmp_path+"/y_test.pkl")
model = load_model(tmp_path + '/3branchMlp_9157.h5')


X_title_test = X_test[[col for col in X_test.columns if col.startswith('title_emb')]]
X_tags_test = X_test[[col for col in X_test.columns if col.startswith('tags_emb')]]
X_desc_test = X_test[[col for col in X_test.columns if col.startswith('desc_emb')]]
X_numeric_test = X_test[['views', 'comment_count', 'engagement_rate', 'like_dislike_ratio', 'tag_count']]


y_pred = model.predict({
    'title_embedding': X_title_test,
    'tags_embedding': X_tags_test,
    'description_embedding': X_desc_test,
    'numeric_features': X_numeric_test
})


y_true_classes = y_test.to_numpy().argmax(axis=1)
y_pred_classes = y_pred.argmax(axis=1)
y_pred_df = pd.DataFrame({'prediction': y_pred_classes})
y_pred_df.to_csv(tmp_path+ "/prediction.csv")
print(classification_report(y_true_classes, y_pred_classes, digits=4))