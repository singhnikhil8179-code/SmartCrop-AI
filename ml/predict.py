from pathlib import Path
import joblib
import pandas as pd
from config import MODEL_PATH

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError("Trained model not found. Run: python ml/train_model.py")
    return joblib.load(MODEL_PATH)

def predict_top_crops(model, values, top_n=3):
    frame = pd.DataFrame([[values[k] for k in FEATURES]], columns=FEATURES)
    probabilities = model.predict_proba(frame)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)[:top_n]
    return [{"crop": str(crop).replace("_", " ").title(),
             "confidence": round(float(prob) * 100, 2)} for crop, prob in ranked]
