# train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Crop_recommendation.csv")


# Model folder & file
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def load_data(path):
    print("Loading dataset from:", path)    # Debug print
    df = pd.read_csv(path)

    # Crop dataset column names
    feature_cols = [c for c in df.columns if c != "label"]
    x = df[feature_cols]
    y = df["label"]
    return x, y, feature_cols

def train():
    x, y, feature_cols = load_data(DATA_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": clf, "features": feature_cols}, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train()