from pathlib import Path
import joblib
import __main__
from app.encoders import CommuneSalesEncoder

#rustine
__main__.CommuneSalesTransformer = CommuneSalesEncoder

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "models"
    / "prix_m2_pipeline_2020.joblib"
)

def get_model():
    return joblib.load(MODEL_PATH)