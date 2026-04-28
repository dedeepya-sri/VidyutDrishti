import joblib
import numpy as np

# Load model
model = joblib.load("../models/model.pkl")

def predict_generation(temp, wind, irr, hum):
    data = np.array([[temp, wind, irr, hum]])
    prediction = model.predict(data)[0]
    return prediction

if __name__ == "__main__":
    pred = predict_generation(32, 6, 850, 40)
    print(f"Predicted Generation: {pred:.2f} MW")