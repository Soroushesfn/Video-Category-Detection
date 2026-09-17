"""Evaluate the trained classifier and save its predictions."""

import pandas as pd
from keras.models import load_model
from sklearn.metrics import classification_report

from paths import MODEL_DIR, TMP_DIR
from train_model import select_inputs


def main() -> None:
    x_test = pd.read_pickle(TMP_DIR / "X_test.pkl")
    y_test = pd.read_pickle(TMP_DIR / "y_test.pkl")
    model_path = MODEL_DIR / "youtube_category_classifier.keras"
    if not model_path.exists():
        # Retain compatibility with the original published checkpoint.
        model_path = MODEL_DIR / "3branchMlp_9157.keras"
    model = load_model(model_path)

    probabilities = model.predict(select_inputs(x_test), verbose=1)
    true_classes = y_test.to_numpy().argmax(axis=1)
    predicted_classes = probabilities.argmax(axis=1)
    pd.DataFrame({"prediction": predicted_classes}).to_csv(
        TMP_DIR / "predictions.csv", index=False
    )
    print(classification_report(true_classes, predicted_classes, digits=4))


if __name__ == "__main__":
    main()
