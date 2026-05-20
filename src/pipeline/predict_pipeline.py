# Import modules
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import dill
# Import libraries
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import functions
from retention_strategy.rentention import retention_profit

# ==============================================================================
# --- Model loading ---
# ==============================================================================

# locate root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# define path
preprocessor_path = ROOT_DIR.joinpath("artifacts/preprocessor/preprocessor.pkl")
model_path = ROOT_DIR.joinpath("models/best_model.pkl")
threshold_path = ROOT_DIR.joinpath("artifacts/threshold/threshold.json")


@asynccontextmanager
async def lifespan(api: FastAPI):
    if not preprocessor_path.exists():
        raise RuntimeError(f"Preprocessor not found: {preprocessor_path}")
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    if not threshold_path.exists():
        raise RuntimeError(f"Threshold not found: {threshold_path}")

    with open(preprocessor_path, "rb") as f:
        api.state.preprocessor = dill.load(f)

    with open(model_path, "rb") as f:
        api.state.model = dill.load(f)

    with open(threshold_path, "r") as f:
        data = json.load(f)
        api.state.threshold = data["best_threshold"]

    yield


# ==============================================================================
# --- 1. Initializing fastapi instance ---
# ==============================================================================

app = FastAPI(
    title="Customer CustomerData Predictor 2026",
    description="Predict customer churn and retention target value",
    version="1.0.0",
    lifespan=lifespan
)


# ==============================================================================
# --- 2. Request schema ---
# ==============================================================================

# create request class
class CustomerData(BaseModel):
    MonthlyCharges: float | int = Field(
        default=100, gt=0, title="Monthly Charges"
    )
    TotalCharges: float | int = Field(
        default=1000, gt=0, title="Total Charges"
    )
    CLTV: float | int = Field(
        default=4226, gt=0, title="CLTV"
    )
    ServiceCount: int = Field(
        default=1, gt=0, title="Service count"
    )
    City: Literal[
        'Los Angeles',
        'San Diego',
        'San Jose',
        'Sacramento',
        'San Francisco',
        'Fresno',
        'Long Beach',
        'Oakland',
        'Stockton',
        'Glendale',
        'Bakersfield',
        'Riverside',
    ] = "Los Angeles"
    Gender: Literal["Male", "Female"] = "Male"
    SeniorCitizen: Literal["Yes", "No"] = "No"
    Partner: Literal["Yes", "No"] = "No"
    Dependents: Literal["No", "Yes"] = "No"
    Contract: Literal[
        'Month-to-month', 'Two year', 'One year'
    ] = "Month-to-month"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"
    PaymentMethod: Literal[
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ] = "Electronic check"
    CustomerLoyalty: Literal[
        "new", "seeker", "floater", "loyal", "ultra_loyal",
    ] = "new"


# create response class
class PredictionResponse(BaseModel):
    model_version: str
    churn_probability: float
    will_churn: bool
    profit: int | float
    should_target: bool


# ==============================================================================
# --- 3. API Endpoints ---
# ==============================================================================

@app.get("/health")
def health():
    return {"status": "Customer Churn Predictor 2026 is up and running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(ch: CustomerData):
    try:
        # Map the training features to the basemodel
        raw_features = {
            "Monthly Charges": ch.MonthlyCharges,
            "Total Charges": ch.TotalCharges,
            "CLTV": ch.CLTV,
            "Service_count": ch.ServiceCount,
            "City": ch.City,
            "Gender": ch.Gender,
            "Senior Citizen": ch.SeniorCitizen,
            "Partner": ch.Partner,
            "Dependents": ch.Dependents,
            "Contract": ch.Contract,
            "Paperless Billing": ch.PaperlessBilling,
            "Payment Method": ch.PaymentMethod,
            "customer_loyalty": ch.CustomerLoyalty
        }

        # Convert to DataFrame
        user_input = pd.DataFrame([raw_features])

        # process the data
        processed_input = app.state.preprocessor.transform(user_input)

        # make prediction
        probability = app.state.model.predict_proba(processed_input)[0, 1]

        # Extract other info
        prediction = bool(int(probability >= app.state.threshold))
        profit = retention_profit(prob=probability)
        should_target = (profit > 0)

        return PredictionResponse(
            model_version="1.0.0",
            churn_probability=round(probability, 4),
            will_churn=prediction,
            profit=round(profit, 2),
            should_target=should_target
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed with exception {e}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "predict_pipeline:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
