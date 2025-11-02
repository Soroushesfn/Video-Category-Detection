import os
import pandas as pd
import tensorflow as tf
from keras import layers, models, Input, saving
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import mlflow
import mlflow.tensorflow

tmp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp"))
X_train = pd.read_pickle(tmp_path+"/X_train.pkl")
X_val = pd.read_pickle(tmp_path+"/X_val.pkl")
y_train = pd.read_pickle(tmp_path+"/y_train.pkl")
y_val = pd.read_pickle(tmp_path+"/y_val.pkl")
cat_count = len([col for col in y_train.columns if col.startswith('cat_')])


X_title_train = X_train[[col for col in X_train.columns if col.startswith('title_emb')]]
X_tags_train = X_train[[col for col in X_train.columns if col.startswith('tags_emb')]]
X_desc_train = X_train[[col for col in X_train.columns if col.startswith('desc_emb')]]
X_numeric_train = X_train[['views', 'comment_count', 'engagement_rate', 'like_dislike_ratio', 'tag_count']]

X_title_val = X_val[[col for col in X_val.columns if col.startswith('title_emb')]]
X_tags_val = X_val[[col for col in X_val.columns if col.startswith('tags_emb')]]
X_desc_val = X_val[[col for col in X_val.columns if col.startswith('desc_emb')]]
X_numeric_val = X_val[['views', 'comment_count', 'engagement_rate', 'like_dislike_ratio', 'tag_count']]



title_input = Input(shape=(384,), name='title_embedding')
tags_input = Input(shape=(384,), name='tags_embedding')
desc_input = Input(shape=(384,), name='description_embedding')
numeric_input = Input(shape=(5,), name='numeric_features')


def embedding_branch(input_tensor):
    x = layers.Dense(128, activation='relu')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    return x


# Embedding path
title_branch = embedding_branch(title_input)
tags_branch = embedding_branch(tags_input)
desc_branch = embedding_branch(desc_input)

# Numeric path
num_branch = layers.Dense(64, activation='relu')(numeric_input)
num_branch = layers.BatchNormalization()(num_branch)
num_branch = layers.Dropout(0.3)(num_branch)

# Fusion
x = layers.concatenate([title_branch, tags_branch, desc_branch, num_branch])
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)

# Output layer
output = layers.Dense(cat_count, activation='softmax', name='category_output')(x)

model = models.Model(inputs=[title_input, tags_input, desc_input, numeric_input], outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

mlflow.set_experiment("Youtube_Category_Predictor")

with mlflow.start_run(run_name="3Branch_MLP") as run:
    # Log parameters
    mlflow.log_param("embedding_dim", 384)
    mlflow.log_param("num_dense_units", 128)
    mlflow.log_param("dropout_rate", 0.3)
    mlflow.log_param("fusion_dense_1", 256)
    mlflow.log_param("fusion_dense_2", 128)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("epochs", 30)

    # Log model structure
    model.summary(print_fn=lambda x: mlflow.log_text(x + "\n", "model_summary.txt"))

    # Enable automatic logging for TensorFlow / Keras
    mlflow.tensorflow.autolog()

history = model.fit(
    {
        'title_embedding': X_title_train,
        'tags_embedding': X_tags_train,
        'description_embedding': X_desc_train,
        'numeric_features': X_numeric_train
    },
    y_train,
    validation_data=(
        {
            'title_embedding': X_title_val,
            'tags_embedding': X_tags_val,
            'description_embedding': X_desc_val,
            'numeric_features': X_numeric_val
        },
        y_val
    ),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop, lr_scheduler]
)
final_val_loss, final_val_acc, final_val_prec, final_val_rec = model.evaluate(
        {
            'title_embedding': X_title_val,
            'tags_embedding': X_tags_val,
            'description_embedding': X_desc_val,
            'numeric_features': X_numeric_val
        },
        y_val, verbose=0)

mlflow.log_metric("val_loss", final_val_loss)
mlflow.log_metric("val_accuracy", final_val_acc)
mlflow.log_metric("val_precision", final_val_prec)
mlflow.log_metric("val_recall", final_val_rec)
mlflow.keras.log_model(model, "3branchMlp_model")

saving.save_model(model, tmp_path + '/3branchMlp_9157.keras')