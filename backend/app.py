from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="VidyutDrishti API")

model = joblib.load("../models/model.pkl")

@app.get("/")
def home():
    return {"message": "VidyutDrishti API Running"}

@app.get("/predict")
def predict(temp: float, wind: float, irr: float, hum: float):
    data = np.array([[temp, wind, irr, hum]])
    prediction = model.predict(data)[0]

    # Simple uncertainty (mock P10/P50/P90)
    return {
        "P50": round(prediction, 2),
        "P10": round(prediction * 0.8, 2),
        "P90": round(prediction * 1.2, 2)
    }