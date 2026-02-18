# asdnrea
import joblib
import numpy as np
from config import DATA_PATH, MODEL_PATH
from data_loader import load_data
from preprocessing import preprocess
from feature_engineering import create_features
from model_rf import build_model

df = load_data(DATA_PATH)
df, scaler = preprocess(df)
df = create_features(df)

X = df[['temperature','vibration','pressure','humidity','temp_vib_ratio']]
y = df['failure']

model = build_model()
model.fit(X, y)

joblib.dump(model, MODEL_PATH)

print("✅ Random Forest Model Trained & Saved")