"""Train the multi-branch category-classification network."""

import mlflow
import mlflow.tensorflow
import pandas as pd
import tensorflow as tf
from keras import Input, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from paths import MODEL_DIR, TMP_DIR, create_runtime_directories


EMBEDDING_DIMENSION = 384
NUMERIC_FEATURES = (
    "views",
    "comment_count",
    "engagement_rate",
    "like_dislike_ratio",
    "tag_count",
)


def select_inputs(dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the feature matrix into the four model branches."""
    return {
        "title_embedding": dataframe.filter(regex=r"^title_emb"),
        "tags_embedding": dataframe.filter(regex=r"^tags_emb"),
        "description_embedding": dataframe.filter(regex=r"^desc_emb"),
        "numeric_features": dataframe.loc[:, NUMERIC_FEATURES],
    }


def build_model(category_count: int) -> models.Model:
    title_input = Input((EMBEDDING_DIMENSION,), name="title_embedding")
    tags_input = Input((EMBEDDING_DIMENSION,), name="tags_embedding")
    description_input = Input((EMBEDDING_DIMENSION,), name="description_embedding")
    numeric_input = Input((len(NUMERIC_FEATURES),), name="numeric_features")

    def embedding_branch(input_tensor):
        branch = layers.Dense(128, activation="relu")(input_tensor)
        branch = layers.BatchNormalization()(branch)
        return layers.Dropout(0.3)(branch)

    numeric_branch = layers.Dense(64, activation="relu")(numeric_input)
    numeric_branch = layers.BatchNormalization()(numeric_branch)
    numeric_branch = layers.Dropout(0.3)(numeric_branch)

    combined = layers.concatenate(
        [
            embedding_branch(title_input),
            embedding_branch(tags_input),
            embedding_branch(description_input),
            numeric_branch,
        ]
    )
    combined = layers.Dense(256, activation="relu")(combined)
    combined = layers.Dropout(0.4)(combined)
    combined = layers.Dense(128, activation="relu")(combined)
    combined = layers.Dropout(0.3)(combined)
    output = layers.Dense(
        category_count, activation="softmax", name="category_output"
    )(combined)

    model = models.Model(
        inputs=[title_input, tags_input, description_input, numeric_input],
        outputs=output,
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main() -> None:
    create_runtime_directories()
    x_train = pd.read_pickle(TMP_DIR / "X_train.pkl")
    x_val = pd.read_pickle(TMP_DIR / "X_val.pkl")
    y_train = pd.read_pickle(TMP_DIR / "y_train.pkl")
    y_val = pd.read_pickle(TMP_DIR / "y_val.pkl")
    model = build_model(y_train.shape[1])

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=4, min_lr=1e-6, verbose=1
        ),
    ]

    mlflow.set_experiment("YouTube_Category_Predictor")
    with mlflow.start_run(run_name="Multi_Branch_MLP"):
        mlflow.log_params(
            {
                "embedding_dimension": EMBEDDING_DIMENSION,
                "batch_size": 64,
                "epochs": 30,
                "random_state": 44,
            }
        )
        mlflow.tensorflow.autolog()
        model.fit(
            select_inputs(x_train),
            y_train,
            validation_data=(select_inputs(x_val), y_val),
            epochs=30,
            batch_size=64,
            callbacks=callbacks,
        )
        loss, accuracy, precision, recall = model.evaluate(
            select_inputs(x_val), y_val, verbose=0
        )
        mlflow.log_metrics(
            {
                "validation_loss": loss,
                "validation_accuracy": accuracy,
                "validation_precision": precision,
                "validation_recall": recall,
            }
        )

        model_path = MODEL_DIR / "youtube_category_classifier.keras"
        model.save(model_path)
        mlflow.log_artifact(str(model_path))
        print(f"[TRAIN] Saved model to {model_path}")


if __name__ == "__main__":
    main()
