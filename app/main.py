from fastapi import FastAPI, Depends
from app.schemas import InputData
from app.model_loader import get_model
from app.security import verify_api_key
import pandas as pd

app = FastAPI()

model = get_model()

@app.get("/")
def root():
    return {"message": "API ML OK 🚀"}

@app.post("/predict")
def predict(data: InputData, api_key: str = Depends(verify_api_key)):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"prix_m2": prediction}