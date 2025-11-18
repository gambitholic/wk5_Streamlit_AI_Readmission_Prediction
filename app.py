import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("readmission_model.pkl")

st.title("🔍 Patient 30-Day Readmission Predictor")
st.write("Enter patient details below.")

# Example input fields (customize based on your dataset)
age = st.selectbox("Age", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
time_in_hospital = st.number_input("Time in hospital (days)", 1, 14)
num_lab_procedures = st.number_input("Number of lab procedures", 0, 100)
num_medications = st.number_input("Number of medications", 0, 50)

# Prepare data for prediction
input_data = pd.DataFrame({
    "age": [age],
    "time_in_hospital": [time_in_hospital],
    "num_lab_procedures": [num_lab_procedures],
    "num_medications": [num_medications]
})

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High Risk of Readmission (Probability: {prob:.2f})")
    else:
        st.success(f"Low Risk of Readmission (Probability: {prob:.2f})")