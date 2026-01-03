# index.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import warnings

# -------------------------------
# Suppress version warnings
# -------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Cardio Disease Prediction API")

# -------------------------------
# Load model and metrics
# -------------------------------
try:
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)
except Exception as e:
    print("❌ Error loading model.pkl:", e)
    data = {}

MODEL = data.get("model", None)
ACCURACY = data.get("accuracy")
TRAIN_ACCURACY = data.get("train_accuracy")
TEST_ACCURACY = data.get("test_accuracy")
PRECISION = data.get("precision")
RECALL = data.get("recall")
F1_SCORE = data.get("f1_score")
BEST_PARAMS = data.get("best_params")

# -------------------------------
# Request body for /predict
# -------------------------------
class CardioInput(BaseModel):
    age: float
    gender: int
    ap_hi: float
    ap_lo: float
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int
    BMI: float

# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
def root():
    acc_text = f"{ACCURACY*100:.2f}%" if ACCURACY is not None else "N/A"
    return {"message": f"Cardio API running. Model accuracy: {acc_text}"}

# -------------------------------
# Predict endpoint
# -------------------------------
@app.post("/predict")
def predict(input_data: CardioInput):
    if MODEL is None:
        return {"error": "Model not loaded. Please check server logs."}

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Predict probability
        prob = MODEL.predict_proba(input_df)[0][1]

        # Determine risk level
        if prob < 0.4:
            risk_level = "Low"
        elif prob < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        return {
            "risk_probability": round(prob*100, 2),
            "risk_level": risk_level
        }

    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

# -------------------------------
# Metrics endpoint
# -------------------------------
@app.get("/metrics")
def metrics():
    return {
        "train_accuracy": TRAIN_ACCURACY,
        "test_accuracy": TEST_ACCURACY,
        "precision": PRECISION,
        "recall": RECALL,
        "f1_score": F1_SCORE,
        "best_params": BEST_PARAMS
    }
