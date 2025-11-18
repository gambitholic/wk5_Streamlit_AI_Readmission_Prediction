import streamlit as st
import joblib
import pandas as pd
from streamlit.components.v1 import html

# ---------------------------
# Load model and column list
# ---------------------------
model = joblib.load("readmission_model.pkl")
columns = joblib.load("model_columns.pkl")

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Patient Readmission Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 30-Day Patient Readmission Predictor")

st.write("This tool uses an AI model trained on real hospital datasets to predict whether a patient will be readmitted within 30 days.")

st.write("---")

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("📋 Patient Information")

age = st.sidebar.selectbox(
    "Age Range",
    ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
     "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
)

time_in_hospital = st.sidebar.number_input(
    "🏥 Time in Hospital (days)", min_value=1, max_value=14, value=3
)

num_lab_procedures = st.sidebar.number_input(
    "🔬 Number of Lab Procedures", min_value=0, max_value=150, value=40
)

num_medications = st.sidebar.number_input(
    "💊 Number of Medications", min_value=0, max_value=70, value=10
)

predict_btn = st.sidebar.button("🔍 Predict Risk", use_container_width=True)

# ---------------------------
# Create full input row matching training structure
# ---------------------------
def build_full_input_row():
    """Create a full dataframe row that contains every training column."""
    df = pd.DataFrame(columns=columns)
    df.loc[0] = [None] * len(columns)

    # Fill only the fields we collect from UI
    df.loc[0, "age"] = age
    df.loc[0, "time_in_hospital"] = time_in_hospital
    df.loc[0, "num_lab_procedures"] = num_lab_procedures
    df.loc[0, "num_medications"] = num_medications

    # For all other missing categorical values, use "Unknown"
    for col in columns:
        if df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    return df

# ---------------------------
# Prediction Logic
# ---------------------------
if predict_btn:
    input_data = build_full_input_row()

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.write("---")

    if prediction == 1:
        st.error(f"❗ *High Risk of Readmission* — Probability: *{prob:.2f}*")
    else:
        st.success(f"✔ *Low Risk of Readmission* — Probability: *{prob:.2f}*")

    # Gauges are optional, include if you want the UI enhancement
    st.write("---")

    gauge_html = f"""
    <div style="width: 100%; text-align: center;">
        <h3>Risk Probability Gauge</h3>
        <svg width="200" height="100">
            <path d="M10 90 A80 80 0 0 1 190 90" fill="none" stroke="#ddd" stroke-width="15" />
            <path d="M10 90 A80 80 0 0 1 {10 + prob*180} 90"
                  fill="none" stroke="#ff4d4d" stroke-width="15" />
            <circle cx="{10 + prob*180}" cy="90" r="8" fill="#ff4d4d" />
        </svg>
        <p style="font-size: 18px;">{prob:.1%} risk</p>
    </div>
    """

    html(gauge_html, height=200)

st.write("---")
st.caption("Developed for AI Development Workflow – Week 5")