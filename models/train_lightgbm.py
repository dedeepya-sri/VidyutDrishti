import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load data
df = pd.read_csv("../data/synthetic_generation_data.csv")

# Features & target
X = df.drop("generation", axis=1)
y = df["generation"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LGBMRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)

print(f"Model trained successfully!")
print(f"MAE: {mae}")

# Save model
joblib.dump(model, "../models/model.pkl")
print("Model saved as model.pkl")