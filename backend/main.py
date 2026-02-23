import os
from pathlib import Path

import bcrypt
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    LoginRequest,
    LoginResponse,
    OrderFeatures,
    PredictionResponse,
    RegisterRequest,
)
from database import create_user, get_user_by_userid, save_prediction, get_recent_predictions


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"

package = None
model = None
encoder = None
threshold = 0.5

if MODEL_PATH.exists():
    package = joblib.load(MODEL_PATH)
    if isinstance(package, dict):
        model = package.get("model")
        encoder = package.get("encoder")
        threshold = float(package.get("threshold", 0.5))
    else:
        model = package


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    if "discount_applied" in df.columns and "product_price" in df.columns:
        denom = df["product_price"].replace(0, np.finfo(float).eps)
        df["discount_pct"] = df["discount_applied"] / denom
    if "product_price" in df.columns and "order_quantity" in df.columns:
        df["order_value"] = df["product_price"] * df["order_quantity"]
    return df


def risk_bucket(probability: float, decision_threshold: float) -> str:
    if probability >= max(decision_threshold + 0.15, 0.75):
        return "High"
    if probability >= decision_threshold:
        return "Medium"
    return "Low"


def recommendation_text(risk_level: str) -> str:
    if risk_level == "High":
        return "Apply proactive intervention: stricter QC, return-policy reminder, and support outreach."
    if risk_level == "Medium":
        return "Review order context and send preventive guidance before fulfillment."
    return "Proceed with standard fulfillment workflow."


app = FastAPI()

# Get allowed origins from environment variable or use defaults
frontend_url = os.getenv("FRONTEND_URL", "")
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
if frontend_url:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: OrderFeatures):
    if model is None:
        return PredictionResponse(
            prediction_label="No",
            probability=0.0,
            decision_threshold=threshold,
            risk_level="Low",
            recommendation="Model not loaded; returning safe default.",
        )

    data = pd.DataFrame([features.dict()])
    data_fe = feature_engineering(data)

    if encoder is not None:
        data_enc = encoder.transform(data_fe)
    else:
        data_enc = data_fe

    proba_array = model.predict_proba(data_enc)
    proba = float(proba_array[0][1])
    prediction = 1 if proba >= threshold else 0
    label = "Yes" if prediction == 1 else "No"

    risk_level = risk_bucket(proba, threshold)
    recommendation = recommendation_text(risk_level)

    save_prediction(features.dict(), prediction, proba)

    return PredictionResponse(
        prediction_label=label,
        probability=proba,
        decision_threshold=threshold,
        risk_level=risk_level,
        recommendation=recommendation,
    )


@app.post("/register")
def register(request: RegisterRequest):
    password_bytes = request.password.encode("utf-8")
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    success = create_user(request.name, request.userid, hashed.decode("utf-8"))
    if not success:
        raise HTTPException(status_code=400, detail="User ID already exists")
    return {"message": "User registered successfully"}


@app.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    user = get_user_by_userid(request.userid)
    if user is None:
        return LoginResponse(success=False, name=None)
    stored_hash = user["password_hash"].encode("utf-8")
    password_bytes = request.password.encode("utf-8")
    if not bcrypt.checkpw(password_bytes, stored_hash):
        return LoginResponse(success=False, name=None)
    return LoginResponse(success=True, name=user["name"])


@app.get("/predictions/recent")
def recent_predictions(limit: int = Query(default=10, ge=1, le=100)):
    rows = get_recent_predictions(limit)
    return rows
