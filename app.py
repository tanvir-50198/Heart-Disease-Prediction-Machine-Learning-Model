import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
from pathlib import Path

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------------------
# Load Model & Dataset
# -------------------------------
BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "model.pkl", "rb") as file:
    model = pk.load(file)

data = pd.read_csv(BASE_DIR / "heart_disease.csv")

# -------------------------------
# App Title
# -------------------------------
st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's information below to predict heart disease risk.")

# -------------------------------
# Gender
# -------------------------------
gender = st.selectbox(
    "Choose Gender",
    data["Gender"].unique()
)

gen = 1 if gender == "Male" else 0

st.divider()

# -------------------------------
# Input Fields
# -------------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=0,
        step=1
    )

    currentSmoker = st.selectbox(
        "Current Smoker",
        ["No", "Yes"]
    )
    currentSmoker = 1 if currentSmoker == "Yes" else 0

    cigsPerDay = st.number_input(
        "Cigarettes Per Day",
        min_value=0
    )

    BPMeds = st.selectbox(
        "BP Medicine",
        ["No", "Yes"]
    )
    BPMeds = 1 if BPMeds == "Yes" else 0

    prevalentStroke = st.selectbox(
        "Previous Stroke",
        ["No", "Yes"]
    )
    prevalentStroke = 1 if prevalentStroke == "Yes" else 0

    prevalentHyp = st.selectbox(
        "Hypertension",
        ["No", "Yes"]
    )
    prevalentHyp = 1 if prevalentHyp == "Yes" else 0

with col2:

    diabetes = st.selectbox(
        "Diabetes",
        ["No", "Yes"]
    )
    diabetes = 1 if diabetes == "Yes" else 0

    totChol = st.number_input(
        "Total Cholesterol",
        min_value=0
    )

    sysBP = st.number_input(
        "Systolic Blood Pressure",
        min_value=0
    )

    diaBP = st.number_input(
        "Diastolic Blood Pressure",
        min_value=0
    )

    BMI = st.number_input(
        "BMI",
        min_value=0.0,
        format="%.2f"
    )

    heartRate = st.number_input(
        "Heart Rate",
        min_value=0
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0
    )

st.write("")

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict", use_container_width=True):

    input_data = np.array([[
        gen,
        age,
        currentSmoker,
        cigsPerDay,
        BPMeds,
        prevalentStroke,
        prevalentHyp,
        diabetes,
        totChol,
        sysBP,
        diaBP,
        BMI,
        heartRate,
        glucose
    ]], dtype=float)

    prediction = model.predict(input_data)

    st.divider()

    if prediction[0] == 0:
        st.success("✅ The patient is NOT at risk of heart disease.")
    else:
        st.error("⚠️ The patient IS at risk of heart disease.")