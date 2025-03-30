import streamlit as st
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import pickle
import datetime
import requests
import os

# Load or initialize user data
USER_DATA_FILE = "users.json"

def load_users():
    try:
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f, indent=4)

users = load_users()

# Streamlit UI
st.set_page_config(page_title="Menstrual Cycle Prediction", layout="wide")
st.title("🌸 Menstrual Cycle Prediction App")

# Authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login():
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success("✅ Login Successful!")
        else:
            st.error("❌ Invalid credentials")

def signup():
    username = st.text_input("🆕 New Username")
    password = st.text_input("🔒 New Password", type="password")
    if st.button("Sign Up"):
        if username in users:
            st.error("❌ Username already exists")
        else:
            users[username] = password
            save_users(users)
            st.success("✅ Account created! Please log in.")

def logout():
    st.session_state["authenticated"] = False
    st.session_state.pop("username", None)
    st.rerun()

if not st.session_state["authenticated"]:
    auth_option = st.radio("🔑 Select option", ["Login", "Sign Up"])
    if auth_option == "Login":
        login()
    else:
        signup()
else:
    st.sidebar.button("🚪 Logout", on_click=logout)
    st.sidebar.write(f"👤 Logged in as: **{st.session_state['username']}**")

    # Load menstrual cycle data
    DATA_FILE = "menstrual_cycle_data_updated.csv"
    if not os.path.exists(DATA_FILE):
        st.error(f"⚠️ Dataset `{DATA_FILE}` not found! Please provide a dataset.")
    else:
        data = pd.read_csv(DATA_FILE)
        
        # Ensure proper date format
        data['Last_Period_Date'] = pd.to_datetime(data['Last_Period_Date'], errors='coerce', dayfirst=True)

        # Ensure all required columns exist
        required_columns = ['Cycle_Length', 'Pain_Level', 'Fatigue_Level', 'Mood_Swings', 'Nausea']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            st.error(f"⚠️ Missing columns in dataset: {missing_cols}")
            st.stop()

        # Scale data
        scaler = MinMaxScaler()
        data[required_columns] = scaler.fit_transform(data[required_columns])

        # Load models
        try:
            lstm_model = load_model("models/lstm_model.keras")  # Ensure correct path
            xgb_model = pickle.load(open("models/xgboost_model.pkl", "rb"))
            st.success("✅ Models loaded successfully!")
        except FileNotFoundError as e:
            st.error(f"❌ Model file missing: {e}")
            st.stop()

        # User Input: Last period date and current period date
        previous_period_date = st.date_input("📅 Enter your last period date:")
        current_period_date = st.date_input("📅 Enter your current period date:")

        if current_period_date <= previous_period_date:
            st.error("❌ Current period date must be later than the last period date.")
        else:
            # Calculate the cycle length
            cycle_length = (current_period_date - previous_period_date).days
            st.write(f"🗓️ Cycle Length: {cycle_length} days")

            # User Symptoms Input (Pain, Fatigue, Mood Swings, Nausea)
            st.subheader("Enter your symptoms:")
            pain_level = st.slider("🤕 Pain Level", 0, 10, 5)
            fatigue_level = st.slider("🤒 Fatigue Level", 0, 10, 5)
            mood_swings = st.slider("🥶 Mood Swings", 0, 10, 5)
            nausea = st.slider("🤢 Nausea", 0, 10, 5)

            # Show the button for prediction
            if st.button("🔮 Predict Next Two Periods"):
                try:
                    # User Input for Prediction
                    user_input = np.array([[cycle_length, pain_level, fatigue_level, mood_swings, nausea]])
                    user_input_scaled = scaler.transform(user_input)  # Scaling for LSTM input

                    # Predict with LSTM
                    lstm_pred = lstm_model.predict(user_input_scaled.reshape(1, 1, 5))
                    predicted_cycle_length = scaler.inverse_transform(
                        np.concatenate([lstm_pred, np.zeros((1, 4))], axis=1)
                    )[0][0]

                    # Predict adjustments with XGBoost
                    xgb_pred = xgb_model.predict(np.array([pain_level, fatigue_level, mood_swings, nausea]).reshape(1, -1))
                    final_cycle_length = predicted_cycle_length + xgb_pred[0]

                    # Calculate the next two period dates
                    next_period_1 = current_period_date + datetime.timedelta(days=int(final_cycle_length))
                    next_period_2 = next_period_1 + datetime.timedelta(days=int(final_cycle_length))

                    # Display results
                    st.success(f"🌸 **Predicted Next Period Date 1: {next_period_1}**")
                    st.success(f"🌸 **Predicted Next Period Date 2: {next_period_2}**")

                except Exception as e:
                    st.error(f"⚠️ Prediction error: {e}")
