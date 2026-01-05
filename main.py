from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import Optional

app = FastAPI(title="Car Price Predictor API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load models - {e}")
    model, scaler, label_encoders = None, None, None


class CarFeatures(BaseModel):
    symboling: int
    fueltype: str
    aspiration: str
    doornumber: int
    carbody: str
    drivewheel: str
    enginelocation: str
    wheelbase: float
    carlength: float
    carwidth: float
    carheight: float
    curbweight: int
    enginetype: str
    cylindernumber: int
    enginesize: int
    fuelsystem: str
    boreratio: float
    stroke: float
    compressionratio: float
    horsepower: int
    peakrpm: int
    citympg: int
    highwaympg: int
    company: str


@app.get("/")
def read_root():
    return {
        "message": "Car Price Predictor API",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Predict car price",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict_price(features: CarFeatures):
    if model is None:
        # Fallback: Simple rule-based prediction
        base_price = 5000
        predicted_price = (
                base_price +
                features.horsepower * 50 +
                features.enginesize * 10 +
                features.curbweight * 2 +
                (10000 if features.company in ['bmw', 'mercedes-benz', 'porsche', 'jaguar'] else 0) +
                (5000 if features.carbody in ['convertible', 'hardtop'] else 0) -
                (features.citympg * 100) +
                (3000 if features.aspiration == 'turbo' else 0)
        )

        return {
            "predicted_price": round(predicted_price, 2),
            "method": "rule_based",
            "breakdown": {
                "base_price": base_price,
                "horsepower_impact": features.horsepower * 50,
                "engine_size_impact": features.enginesize * 10,
                "weight_impact": features.curbweight * 2,
                "brand_premium": 10000 if features.company in ['bmw', 'mercedes-benz', 'porsche', 'jaguar'] else 0,
                "body_type_premium": 5000 if features.carbody in ['convertible', 'hardtop'] else 0
            }
        }

    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([features.dict()])

        # Encode categorical variables
        for col in ['fueltype', 'aspiration', 'carbody', 'drivewheel',
                    'enginelocation', 'enginetype', 'fuelsystem', 'company']:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Make prediction
        predicted_price = model.predict(input_scaled)[0]

        return {
            "predicted_price": round(float(predicted_price), 2),
            "method": "ml_model"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
