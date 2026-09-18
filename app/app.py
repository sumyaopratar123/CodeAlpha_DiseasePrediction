import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(
    page_title="Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")

st.title("🩺 Disease Prediction from Medical Data")
st.caption("CodeAlpha Machine Learning Internship — Task 4")

st.warning(
    "Educational/research use only. Predictions are not medical diagnoses "
    "and should not replace evaluation by a qualified healthcare professional."
)

# ---------- Model helpers ----------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

def available_models(prefix):
    names = ["logistic_regression", "svm", "random_forest", "xgboost"]
    return {
        n: os.path.join(MODEL_DIR, f"{prefix}_{n}.joblib")
        for n in names
        if os.path.exists(os.path.join(MODEL_DIR, f"{prefix}_{n}.joblib"))
    }

def predict(prefix, values):
    models = available_models(prefix)
    if not models:
        st.error("Models are not available. Run `python src/train_all.py` first.")
        return

    outputs = []
    x = pd.DataFrame([values])

    for key, path in models.items():
        model = load_model(path)
        pred = int(model.predict(x)[0])
        prob = float(model.predict_proba(x)[0, 1])
        outputs.append({
            "Model": key.replace("_", " ").title(),
            "Prediction": "Positive" if pred == 1 else "Negative",
            "Probability": f"{prob * 100:.2f}%",
            "_prob": prob,
        })

    out = pd.DataFrame(outputs)
    display_out = out[["Model", "Prediction", "Probability"]].copy()
    st.subheader("Prediction Results")
    st.dataframe(display_out, use_container_width=True, hide_index=True)

# ---------- Header ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 10px; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "❤️ Heart Disease",
    "🩸 Diabetes",
    "🎗️ Breast Cancer",
])

# ============================================================
# HEART DISEASE
# ============================================================
with tabs[0]:
    st.subheader("❤️ Heart Disease Prediction")
    st.write("Enter the clinical measurements used by the UCI Heart Disease dataset.")

    vals = {}

    st.markdown("#### 👤 Patient Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        vals["age"] = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
    with c2:
        sex_label = st.selectbox("Sex", ["Male", "Female"])
        vals["sex"] = 1 if sex_label == "Male" else 0
    with c3:
        cp_label = st.selectbox(
            "Chest Pain Type",
            [
                "Typical angina",
                "Atypical angina",
                "Non-anginal pain",
                "Asymptomatic",
            ],
        )
        vals["cp"] = {
            "Typical angina": 0,
            "Atypical angina": 1,
            "Non-anginal pain": 2,
            "Asymptomatic": 3,
        }[cp_label]

    st.markdown("#### 🩺 Clinical Measurements")
    c1, c2, c3 = st.columns(3)
    with c1:
        vals["trestbps"] = st.number_input(
            "Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130, step=1
        )
    with c2:
        vals["chol"] = st.number_input(
            "Serum Cholesterol (mg/dL)", min_value=50, max_value=700, value=240, step=1
        )
    with c3:
        fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        vals["fbs"] = 1 if fbs_label == "Yes" else 0

    st.markdown("#### 🫀 Heart Test Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        restecg_label = st.selectbox(
            "Resting ECG",
            ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"],
        )
        vals["restecg"] = {
            "Normal": 0,
            "ST-T wave abnormality": 1,
            "Left ventricular hypertrophy": 2,
        }[restecg_label]
    with c2:
        vals["thalach"] = st.number_input(
            "Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150, step=1
        )
    with c3:
        exang_label = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        vals["exang"] = 1 if exang_label == "Yes" else 0

    st.markdown("#### 📊 Additional Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        vals["oldpeak"] = st.number_input(
            "ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1
        )
    with c2:
        slope_label = st.selectbox(
            "ST Segment Slope",
            ["Upsloping", "Flat", "Downsloping"],
        )
        vals["slope"] = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope_label]
    with c3:
        vals["ca"] = st.selectbox(
            "Major Vessels (0–3)", [0, 1, 2, 3], index=0
        )
    with c4:
        thal_label = st.selectbox(
            "Thalassemia Test",
            ["Normal", "Fixed defect", "Reversible defect"],
        )
        # UCI processed data convention: 3=normal, 6=fixed defect, 7=reversible defect.
        vals["thal"] = {
            "Normal": 3,
            "Fixed defect": 6,
            "Reversible defect": 7,
        }[thal_label]

    st.divider()
    if st.button("🔍 Predict Heart Disease", type="primary", use_container_width=True):
        predict("heart", vals)

# ============================================================
# DIABETES
# ============================================================
with tabs[1]:
    st.subheader("🩸 Diabetes Prediction")
    st.write("Enter the patient measurements used by the Pima Indians Diabetes dataset.")

    vals = {}

    st.markdown("#### 👤 Patient Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        vals["Pregnancies"] = st.number_input("Pregnancies", 0, 20, 1, 1)
    with c2:
        vals["Age"] = st.number_input("Age (years)", 1, 120, 30, 1)
    with c3:
        vals["BMI"] = st.number_input("BMI", 0.0, 80.0, 25.0, 0.1)
    with c4:
        vals["DiabetesPedigreeFunction"] = st.number_input(
            "Diabetes Pedigree Function", 0.0, 3.0, 0.47, 0.01
        )

    st.markdown("#### 🧪 Blood & Clinical Measurements")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        vals["Glucose"] = st.number_input("Glucose (mg/dL)", 0, 300, 120, 1)
    with c2:
        vals["BloodPressure"] = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70, 1)
    with c3:
        vals["SkinThickness"] = st.number_input("Skin Thickness (mm)", 0, 100, 20, 1)
    with c4:
        vals["Insulin"] = st.number_input("Insulin (µU/mL)", 0, 1000, 80, 1)

    st.divider()
    if st.button("🔍 Predict Diabetes", type="primary", use_container_width=True):
        predict("diabetes", vals)

# ============================================================
# BREAST CANCER
# ============================================================
with tabs[2]:
    st.subheader("🎗️ Breast Cancer Prediction")
    st.write(
        "Enter the 30 diagnostic measurements used by the Breast Cancer Wisconsin "
        "(Diagnostic) dataset."
    )
    st.info("Use the model's expected numeric feature values. This is an educational demo, not a diagnostic tool.")

    feature_path = os.path.join(MODEL_DIR, "breastcancer_features.json")
    if os.path.exists(feature_path):
        with open(feature_path, encoding="utf-8") as f:
            features = json.load(f)

        vals = {}
        # Group the 30 features into logical sections for a compact UI.
        # The sklearn dataset is bundled with scikit-learn, so it can also
        # provide sensible demo defaults without downloading another file.
        breast_data = load_breast_cancer(as_frame=True).data
        medians = breast_data.median()

        groups = {
            "Mean Measurements": [f for f in features if f.startswith("mean ")],
            "Standard Error Measurements": [f for f in features if f.endswith(" error")],
            "Worst Measurements": [f for f in features if f.startswith("worst ")],
        }

        for group_name, group_features in groups.items():
            with st.expander(group_name, expanded=(group_name == "Mean Measurements")):
                cols = st.columns(3)
                for i, feature in enumerate(group_features):
                    with cols[i % 3]:
                        default_value = float(medians[feature]) if feature in medians else 0.0
                        vals[feature] = st.number_input(
                            feature.title(),
                            value=default_value,
                            step=max(default_value / 100, 0.001),
                            format="%.4f",
                            key=f"bc_{feature}",
                            help=f"Model feature: {feature}",
                        )

        # Safety check: preserve exact training feature order.
        vals = {feature: vals.get(feature, 0.0) for feature in features}

        st.divider()
        if st.button("🔍 Predict Breast Cancer", type="primary", use_container_width=True):
            predict("breastcancer", vals)
    else:
        st.error("Breast cancer feature metadata not found. Run `python src/train_all.py` first.")

# ---------- Footer ----------
st.divider()
st.markdown(
    """
    **Algorithms:** Logistic Regression • SVM • Random Forest • XGBoost

    **Disclaimer:** This application is intended for educational/research purposes only.
    It does not provide medical diagnosis or treatment advice.
    """
)

st.markdown(
   
    """
     ------------------------------------------------------------About------------------------------------------------------------------

    **Contact:** 24/7 
    
    **support:** sumitade324@gmail.com
    """
)
