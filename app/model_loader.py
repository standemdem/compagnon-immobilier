from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.joblib"

def get_model():
    return joblib.load(MODEL_PATH) 