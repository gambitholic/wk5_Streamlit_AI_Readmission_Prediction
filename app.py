import streamlit as st
import pandas as pd
import joblib
from streamlit.components.v1 import html

# ==============================
# LOAD MODEL AND METADATA
# ==============================
model = joblib.load("readmission_model.pkl")
columns = joblib.load("model_columns.pkl")
dtypes = joblib.load("model_dtypes.pkl")

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Patient Readmission Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Patient 30-Day Readmission Risk Predictor")
st.write("Predict whether a patient is at risk of being readmitted within 30 days after discharge.")

st.write("---")

# ==============================
# SIDEBAR INPUTS
# ==============================
st.sidebar.header("📋 Patient Input Fields")

age = st.sidebar.selectbox(
    "Age Range",
    ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
     "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
)

time_in_hospital = st.sidebar.number_input(
    "🏥 Time in Hospital (days)",
    min_value=1, max_value=14, value=3
)

num_lab_procedures = st.sidebar.number_input(
    "🔬 Number of Lab Procedures",
    min_value=0, max_value=150, value=40
)

num_medications = st.sidebar.number_input(
    "💊 Number of Medications",
    min_value=0, max_value=70, value=10
)

predict_btn = st.sidebar.button("🔍 Predict", use_container_width=True)

# ==============================
# BUILD FULL INPUT ROW
# ==============================
def build_input_row():
    """
    Reconstruct a full feature vector matching training columns,
    with correct order and correct dtypes.
    """

    # Create empty row with all training columns
    df = pd.DataFrame({col: [None] for col in columns})

    # Insert the user inputs
    df.loc[0, "age"] = age
    df.loc[0, "time_in_hospital"] = time_in_hospital
    df.loc[0, "num_lab_procedures"] = num_lab_procedures
    df.loc[0, "num_medications"] = num_medications

    # Fill missing values based on dtype
    for col in columns:
        if pd.isna(df.loc[0, col]):
            if dtypes[col] == "object":
                df.loc[0, col] = "Unknown"
            else:
                df.loc[0, col] = 0

    # Enforce correct data types
    for col in df.columns:
        if dtypes[col] == "object":
            df[col] = df[col].astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

# ==============================
# RUN PREDICTION
# ==============================
if predict_btn:
    input_data = build_input_row()

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.write("---")

    if prediction == 1:
        st.error(f"❗ High Risk of Readmission — Probability: *{prob:.2f}*")
    else:
        st.success(f"✔ Low Risk of Readmission — Probability: *{prob:.2f}*")

    # Probability Gauge
    gauge_html = f"""
    <div style="width:100%; text-align:center;">
        <h3>Risk Probability Gauge</h3>
        <svg width="220" height="110">
            <path d="M10 100 A90 90 0 0 1 210 100" fill="none" stroke="#eee" stroke-width="18" />
            <path d="M10 100 A90 90 0 0 1 {10 + prob*200} 100"
                  fill="none" stroke="#ff4d4d" stroke-width="18" />
            <circle cx="{10 + prob*200}" cy="100" r="10" fill="#ff4d4d" />
        </svg>
        <p style="font-size:18px;font-weight:bold;">{prob:.1%} Risk</p>
    </div>
    """
    html(gauge_html, height=180)

st.write("---")
st.caption("Developed for AI Development Workflow – Week 5")