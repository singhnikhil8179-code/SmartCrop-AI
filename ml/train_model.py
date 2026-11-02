"""Train and compare multiple classifiers; save the best model and metrics."""
from pathlib import Path
import json
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DATA_PATH, MODEL_PATH, MODEL_INFO_PATH

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset missing. First run: python ml/fetch_data.py")
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df[required].dropna()
    X, y = df[FEATURES], df[TARGET].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "K-Nearest Neighbours": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Gaussian Naive Bayes": GaussianNB()
    }

    results = {}
    best_name, best_model, best_accuracy = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision_weighted": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4)
        }
        results[name] = metrics
        print(f"{name}: {metrics}")
        if metrics["accuracy"] > best_accuracy:
            best_name, best_model, best_accuracy = name, model, metrics["accuracy"]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    # Save a report from the untouched test split.
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix – {best_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    chart_path = ROOT / "static" / "images" / "confusion_matrix.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_path, dpi=160)
    plt.close()

    info = {
        "best_model": best_name,
        "dataset_rows_after_cleaning": int(len(df)),
        "number_of_crops": int(y.nunique()),
        "features": FEATURES,
        "metrics": results,
        "best_model_classification_report": report,
        "crop_labels": labels,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42,
        "test_size": 0.20
    }
    MODEL_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nBest model: {best_name} ({best_accuracy:.2%} test accuracy)")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics saved: {MODEL_INFO_PATH}")

if __name__ == "__main__":
    main()
