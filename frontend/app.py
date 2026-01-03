import streamlit as st
import requests
import os
# -------------------------------
# FastAPI URL (LOCAL or DEPLOY)
# -------------------------------
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/predict")

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="Cardio Disease Prediction",
    layout="centered"
)

st.title("❤️ Cardio Disease Prediction App")
st.write("Predict the risk of cardiovascular disease using a Machine Learning model.")
st.markdown("---")

# -------------------------------
# Input Form
# -------------------------------
with st.form("input_form"):
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ap_hi = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=200, value=80)
    cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
    gluc = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
    smoke = st.selectbox("Smoker", ["No", "Yes"])
    alco = st.selectbox("Alcohol Intake", ["No", "Yes"])
    active = st.selectbox("Physically Active", ["No", "Yes"])
    height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight_kg = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)

    submit = st.form_submit_button("Predict")

# -------------------------------
# Prediction Logic
# -------------------------------
if submit:
    if ap_lo >= ap_hi:
        st.error("❌ Diastolic BP must be lower than Systolic BP")
        st.stop()

    # Encode inputs
    gender_val = 1 if gender == "Male" else 2
    smoker = 1 if smoke == "Yes" else 0
    alcoholic = 1 if alco == "Yes" else 0
    physically_active = 1 if active == "Yes" else 0

    chol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}

    # BMI calculation
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 2)
    bmi = max(10, min(60, bmi))

    # Prepare payload
    payload = {
        "age": float(age),
        "gender": gender_val,
        "ap_hi": float(ap_hi),
        "ap_lo": float(ap_lo),
        "cholesterol": chol_map[cholesterol],
        "gluc": gluc_map[gluc],
        "smoke": smoker,
        "alco": alcoholic,
        "active": physically_active,
        "BMI": bmi
    }

    # API Call
    try:
        response = requests.post(FASTAPI_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.subheader("🧠 Prediction Result")
            st.metric("Risk Probability", f"{result['risk_probability']}%")

            if result["risk_level"] == "Low":
                st.success("✅ Low risk of cardiovascular disease")
            elif result["risk_level"] == "Moderate":
                st.warning("⚠️ Moderate risk of cardiovascular disease")
            else:
                st.error("🚨 High risk of cardiovascular disease")
        else:
            st.error("❌ Backend error. Please try again later.")

    except Exception as e:
        st.error(f"❌ Unable to connect to backend API: {e}")
# -------------------------------
# Model Performance Visualization
# -------------------------------
st.markdown("---")
st.subheader("📊 Model Performance Analysis")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.image(
    os.path.join(BASE_DIR, "confusion_matrix.png"),
    caption="Confusion Matrix",
    use_container_width=True
)

st.image(
    os.path.join(BASE_DIR, "metrics_bar.png"),
    caption="Performance Metrics",
    use_container_width=True
)

st.image(
    os.path.join(BASE_DIR, "roc_curve.png"),
    caption="ROC Curve",
    use_container_width=True
)


st.markdown("---")
st.caption("Developed by Aum | Streamlit + FastAPI ML Project")

