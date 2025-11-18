import streamlit as st
import pandas as pd
import joblib
from streamlit.components.v1 import html
import numpy as np

# ==============================
# LOAD MODEL AND METADATA
# ==============================
# In a real app, wrap these in a try/except block for better error handling 
# if the files might not be present.
try:
    model = joblib.load("readmission_model.pkl")
    columns = joblib.load("model_columns.pkl")
    dtypes = joblib.load("model_dtypes.pkl")
except FileNotFoundError as e:
    st.error(f"Required model files not found: {e}. Please ensure 'readmission_model.pkl', 'model_columns.pkl', and 'model_dtypes.pkl' are in the directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()


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

# --- Essential Inputs (Numeric and Age) ---
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

# --- Additional Inputs (Categorical) ---
# Adding these greatly improves robustness and prediction quality
race = st.sidebar.selectbox(
    "👤 Race",
    ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"]
)

gender = st.sidebar.selectbox(
    "🚻 Gender",
    ["Female", "Male", "Other"]
)

a1c_result = st.sidebar.selectbox(
    "🩸 A1C Result",
    ["None", ">8", ">7", "Norm"]
)

predict_btn = st.sidebar.button("🔍 Predict", use_container_width=True)

# ==============================
# BUILD FULL INPUT ROW
# ==============================
def build_input_row():
    """
    Reconstruct a full feature vector matching training columns,
    with correct order and correct dtypes.
    
    CRITICAL FIX: Ensure all categorical columns not explicitly set 
    are defaulted to a value known to the model (e.g., 'Caucasian', 'None').
    """

    # Create empty row with all training columns
    df = pd.DataFrame({col: [np.nan] for col in columns}) # Use np.nan for clearer handling

    # --- 1. Insert the user inputs ---
    df.loc[0, "age"] = age
    df.loc[0, "time_in_hospital"] = time_in_hospital
    df.loc[0, "num_lab_procedures"] = num_lab_procedures
    df.loc[0, "num_medications"] = num_medications
    
    # Insert new user inputs
    if "race" in columns:
        df.loc[0, "race"] = race
    if "gender" in columns:
        df.loc[0, "gender"] = gender
    if "A1Cresult" in columns: # Check the exact column name used in training
        df.loc[0, "A1Cresult"] = a1c_result

    # --- 2. Fill missing values with safe defaults ---
    for col in columns:
        if pd.isna(df.loc[0, col]):
            if dtypes.get(col) == "object":
                # Use a specific, known, safe default for categorical features.
                # 'Unknown' or '?' is only safe if it was in the training data.
                # Assuming 'None' or 'Caucasian' are common modes.
                safe_default = "None" 
                if col == 'race':
                    safe_default = 'Caucasian'
                elif col == 'gender':
                    safe_default = 'Female'
                
                df.loc[0, col] = safe_default
            else:
                # Use 0 for missing numeric features
                df.loc[0, col] = 0

    # --- 3. Enforce correct data types and column order ---
    df = df[columns] # Ensure correct column order is maintained
    
    for col in df.columns:
        target_dtype = dtypes.get(col, 'object') # Use 'object' if type is unknown

        if target_dtype == "object":
            df[col] = df[col].astype(str)
        else:
            # Use pd.to_numeric and fill any errors (like NaNs) with 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(target_dtype)

    return df

# ==============================
# RUN PREDICTION
# ==============================
if predict_btn:
    input_data = build_input_row()

    try:
        # Check the shape and columns before passing to the model (for debugging)
        # st.write("Input Data Shape:", input_data.shape)
        # st.write("Input Data Columns:", list(input_data.columns))
        
        prediction = model.predict(input_data)[0]
        # model.predict_proba returns an array, we take the probability of the positive class (index 1)
        prob = model.predict_proba(input_data)[0][1] 

        st.write("---")

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🔴 **High Risk of Readmission** — Probability: *{prob:.2f}*")
            st.warning("The model suggests this patient is likely to be readmitted within 30 days. Recommend follow-up care and patient education.")
        else:
            st.success(f"🟢 **Low Risk of Readmission** — Probability: *{prob:.2f}*")
            st.info("The model suggests this patient is unlikely to be readmitted within 30 days. Continue standard discharge protocol.")

        # Probability Gauge
        gauge_html = f"""
        <div style="width:100%; text-align:center;">
            <h3 style="margin-top: 20px;">Risk Probability Gauge</h3>
            <svg width="220" height="110" viewBox="0 0 220 110">
                <!-- Background Arc -->
                <path d="M10 100 A90 90 0 0 1 210 100" fill="none" stroke="#e0e0e0" stroke-width="18" />
                <!-- Foreground Arc based on probability -->
                <path id="riskArc" d="M10 100 A90 90 0 0 1 {10 + prob*200} 100"
                      fill="none" stroke="#ff4d4d" stroke-width="18" 
                      stroke-linecap="round" />
                <!-- Needle/Indicator -->
                <circle cx="{10 + prob*200}" cy="100" r="10" fill="#ff4d4d" stroke="#fff" stroke-width="2" />
            </svg>
            <p style="font-size:18px;font-weight:bold; color: #333;">{prob:.1%} Risk</p>
        </div>
        """
        html(gauge_html, height=180)

    except ValueError as e:
        # Catch the specific ValueError related to unseen data or pipeline issue
        st.error(f"Prediction Error: The model encountered an issue processing the input data. This usually means a feature name or categorical value is not recognized by the pre-processing pipeline. Detailed error: {e}")
        st.error("Please verify that all categorical defaults are known categories from the training set.")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")


st.write("---")
st.caption("Developed for AI Development Workflow By Wanjiru Ian – Week 5")