import joblib

def get_model():
    return joblib.load("model/model.joblib")