import streamlit as st
import joblib
import pandas as pd
from streamlit.components.v1 import html

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("readmission_model.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Patient Readmission Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# -----------------------------
# Custom CSS (optional)
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #F6F9FC;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }
    .risk-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .high-risk {
        background-color: #ffe5e5;
        border-left: 8px solid #ff4d4d;
    }
    .low-risk {
        background-color: #e5ffe5;
        border-left: 8px solid #33cc33;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📋 Patient Information")

age = st.sidebar.selectbox(
    "Age Range",
    ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
     "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
)

time_in_hospital = st.sidebar.number_input(
    "🏥 Time in Hospital (days)", 1, 14
)

num_lab_procedures = st.sidebar.number_input(
    "🔬 Number of Lab Procedures", 0, 150
)

num_medications = st.sidebar.number_input(
    "💊 Number of Medications", 0, 70
)

st.sidebar.write("----")
predict_btn = st.sidebar.button("🔍 Predict Readmission Risk", use_container_width=True)

# -----------------------------
# Main Page Title
# -----------------------------
st.title("🩺 Patient 30-Day Readmission Risk Prediction")
st.write("This AI system predicts the likelihood a patient will be readmitted within 30 days of hospital discharge.")

st.write("----")

# -----------------------------
# Input Data Preparation
# -----------------------------
input_data = pd.DataFrame({
    "age": [age],
    "time_in_hospital": [time_in_hospital],
    "num_lab_procedures": [num_lab_procedures],
    "num_medications": [num_medications]
})

# -----------------------------
# Display Input Summary Cards
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Age Group", age)
col2.metric("Hospital Stay", f"{time_in_hospital} days")
col3.metric("Lab Procedures", num_lab_procedures)
col4.metric("Medications", num_medications)

st.write("----")

# -----------------------------
# Prediction Logic
# -----------------------------
if predict_btn:
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # ---- Risk Box ----
    if prediction == 1:
        st.markdown(
            f"""
            <div class="risk-box high-risk">
                <h2>❗ High Readmission Risk</h2>
                <h3>Probability: {prob:.2f}</h3>
                <p>The patient has a significant risk of being readmitted within 30 days.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="risk-box low-risk">
                <h2>✔ Low Readmission Risk</h2>
                <h3>Probability: {prob:.2f}</h3>
                <p>The patient is unlikely to be readmitted within 30 days.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------
    # Probability Gauge Chart
    # -------------------------
    gauge_html = f"""
    <div style="width: 100%; text-align: center; margin-top: 40px;">
        <h3>Risk Probability Gauge</h3>
        <svg width="200" height="100">
            <path d="M10 90 A80 80 0 0 1 190 90" fill="none" stroke="#ccc" stroke-width="15" />
            <path d="M10 90 A80 80 0 0 1 {10 + prob*180} 90"
                  fill="none" stroke="#ff4d4d" stroke-width="15" />
            <circle cx="{10 + prob*180}" cy="90" r="8" fill="#ff4d4d" />
        </svg>
        <p style="font-size: 18px;">{prob:.1%} risk</p>
    </div>
    """

    html(gauge_html, height=200)

# -----------------------------
# Footer
# -----------------------------
st.write("----")
st.caption("Developed for AI Development Workflow – Week 5 · Streamlit Deployment Project")