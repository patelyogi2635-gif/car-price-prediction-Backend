from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Car Price Predictor API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict car price",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "method": "rule_based_prediction"
    }


@app.post("/predict")
def predict_price(features: CarFeatures):
    """
    Predict car price using rule-based algorithm
    """
    try:
        # Base price
        base_price = 5000
        
        # Calculate price based on features
        price = base_price
        
        # Horsepower impact (most significant)
        price += features.horsepower * 50
        
        # Engine size impact
        price += features.enginesize * 10
        
        # Weight impact
        price += features.curbweight * 2
        
        # Brand premium
        luxury_brands = ['bmw', 'mercedes-benz', 'porsche', 'jaguar', 'audi']
        if features.company.lower() in luxury_brands:
            price += 10000
        
        # Body type premium
        premium_bodies = ['convertible', 'hardtop']
        if features.carbody in premium_bodies:
            price += 5000
        
        # Turbo premium
        if features.aspiration == 'turbo':
            price += 3000
        
        # Cylinder impact
        price += features.cylindernumber * 500
        
        # MPG impact (better efficiency slightly reduces price for economy cars)
        price -= features.citympg * 100
        
        # Drive wheel premium
        if features.drivewheel == '4wd':
            price += 2000
        elif features.drivewheel == 'rwd':
            price += 1000
        
        # Round to 2 decimal places
        predicted_price = round(price, 2)
        
        # Ensure minimum price
        if predicted_price < 3000:
            predicted_price = 3000
        
        # Calculate breakdown
        breakdown = {
            "base_price": base_price,
            "horsepower_impact": features.horsepower * 50,
            "engine_size_impact": features.enginesize * 10,
            "weight_impact": features.curbweight * 2,
            "brand_premium": 10000 if features.company.lower() in luxury_brands else 0,
            "body_type_premium": 5000 if features.carbody in premium_bodies else 0,
            "turbo_premium": 3000 if features.aspiration == 'turbo' else 0,
            "cylinder_impact": features.cylindernumber * 500,
            "mpg_adjustment": -features.citympg * 100,
            "drivetrain_premium": 2000 if features.drivewheel == '4wd' else (1000 if features.drivewheel == 'rwd' else 0)
        }
        
        return {
            "predicted_price": predicted_price,
            "method": "rule_based",
            "breakdown": breakdown,
            "confidence": "high"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
