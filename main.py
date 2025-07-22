import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Load model and scaler
MODEL_DIR = "measles_model_output"
model = joblib.load(os.path.join(MODEL_DIR, "measles_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

st.set_page_config(page_title="Measles Prediction System", layout="centered")

st.title("🧪 Measles Prediction System")
st.markdown("Fill the form below to predict the likelihood of measles.")

# Input Form
with st.form("measles_form"):
    age = st.slider("Age", 0, 100, 25)
    fever = st.slider("Fever (°C)", 36.0, 42.0, 38.5)
    rash_duration = st.slider("Rash Duration (days)", 0, 10, 2)
    cough = st.selectbox("Cough", ["No", "Yes"])
    conjunctivitis = st.selectbox("Conjunctivitis", ["No", "Yes"])
    koplik_spots = st.selectbox("Koplik Spots", ["No", "Yes"])
    immunization_status = st.selectbox("Immunized?", ["No", "Yes"])
    submitted = st.form_submit_button("Predict")

if submitted:
    # Map inputs
    input_data = np.array([[
        age,
        fever,
        rash_duration,
        1 if cough == "Yes" else 0,
        1 if conjunctivitis == "Yes" else 0,
        1 if koplik_spots == "Yes" else 0,
        1 if immunization_status == "Yes" else 0
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    result = "🟢 No Measles Detected" if prediction == 0 else "🔴 Measles Detected"
    st.success(f"Prediction Result: **{result}**")

    # Show prediction confidence
    proba = model.predict_proba(input_scaled)[0]
    st.info(f"Confidence: Measles: {proba[1]:.2%} | No Measles: {proba[0]:.2%}")

# Sidebar
st.sidebar.title("📊 Model Evaluation")
with open(os.path.join(MODEL_DIR, "evaluation_metrics.txt"), "r") as f:
    metrics = f.read()
st.sidebar.text_area("Metrics Summary", value=metrics, height=300)

