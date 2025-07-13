# app.py

import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("diabetes_model.sav")
scaler = joblib.load("scaler.sav")

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")
st.title("🩺 Diabetes Prediction Web App")
st.write("Enter your health details below to check your diabetes risk:")

# Input form
Pregnancies = st.number_input("Number of Pregnancies", min_value=0)
Glucose = st.number_input("Glucose Level", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin Level", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    input_data = (
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age,
    )

    input_np = np.asarray(input_data).reshape(1, -1)
    std_input = scaler.transform(input_np)
    prediction = model.predict(std_input)

    if prediction[0] == 0:
        st.success("✅ The person is Non-Diabetic")
    else:
        st.error("⚠️ The person is Diabetic")
