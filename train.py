import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_NAME = "target"


def train_and_save():
    csv_path = os.path.join("data", "heart.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model trained successfully. Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds))

    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "heart_model.joblib")

    model_artifact = {
        "model": clf,
        "features": FEATURE_NAMES,
        "model_type": "RandomForestClassifier",
        "accuracy": float(acc),
    }

    joblib.dump(model_artifact, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_and_save()
