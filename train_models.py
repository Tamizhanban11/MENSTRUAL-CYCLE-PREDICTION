import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Generate synthetic data (Replace with actual menstrual cycle dataset)
np.random.seed(42)
cycle_data = np.random.randint(25, 35, size=(100, 1))  # Simulated cycle lengths
symptoms_data = np.random.rand(100, 5)  # 5 random symptom features
labels = cycle_data.flatten() + np.random.randint(-2, 3, size=100)  # Flatten labels to 1D

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(symptoms_data, labels, test_size=0.2, random_state=42)

# 🔹 Train LSTM Model
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # Shape (samples, time_steps=1, features)
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, 5)),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=10, verbose=1)

# Save LSTM Model
lstm_model.save("models/lstm_model.keras")
print("✅ LSTM Model trained and saved as 'models/lstm_model.keras'!")

# 🔹 Train XGBoost Model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)

# Save XGBoost Model
joblib.dump(xgb_model, "models/xgboost_model.pkl")
print("✅ XGBoost model trained and saved as 'models/xgboost_model.pkl'!")

# Evaluate XGBoost Model
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"📊 XGBoost Model Test MSE: {mse:.4f}")
