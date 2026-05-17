from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.retention_strategy.rentention import retention_profit


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
PREPROCESSOR_PATH = ROOT_DIR / "artifacts" / "preprocessor" / "preprocessor.pkl"

SERVICE_COLUMNS = [
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]


class CustomerData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    city: str = Field(..., alias="City", examples=["Los Angeles"])
    gender: Literal["Male", "Female"] = Field(..., alias="Gender")
    senior_citizen: Literal["Yes", "No"] = Field(..., alias="Senior Citizen")
    partner: Literal["Yes", "No"] = Field(..., alias="Partner")
    dependents: Literal["Yes", "No"] = Field(..., alias="Dependents")
    tenure_months: int = Field(..., alias="Tenure Months", ge=0, le=72)
    phone_service: Literal["Yes", "No"] = Field(..., alias="Phone Service")
    multiple_lines: str = Field(..., alias="Multiple Lines", examples=["No"])
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., alias="Internet Service"
    )
    online_security: str = Field(..., alias="Online Security", examples=["Yes"])
    online_backup: str = Field(..., alias="Online Backup", examples=["Yes"])
    device_protection: str = Field(..., alias="Device Protection", examples=["No"])
    tech_support: str = Field(..., alias="Tech Support", examples=["No"])
    streaming_tv: str = Field(..., alias="Streaming TV", examples=["No"])
    streaming_movies: str = Field(..., alias="Streaming Movies", examples=["No"])
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., alias="Contract"
    )
    paperless_billing: Literal["Yes", "No"] = Field(..., alias="Paperless Billing")
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(..., alias="Payment Method")
    monthly_charges: float = Field(..., alias="Monthly Charges", ge=0)
    total_charges: float = Field(..., alias="Total Charges", ge=0)
    cltv: int = Field(..., alias="CLTV", ge=0)


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    retention_profit: float
    should_target: bool


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability and retention targeting value.",
    version="1.0.0",
)


@app.on_event("startup")
def load_artifacts() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise RuntimeError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")

    app.state.model = joblib.load(MODEL_PATH)
    app.state.preprocessor = joblib.load(PREPROCESSOR_PATH)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Customer churn prediction API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData) -> PredictionResponse:
    try:
        input_df = prepare_features(customer)
        processed_input = app.state.preprocessor.transform(input_df)

        probability = float(app.state.model.predict_proba(processed_input)[0, 1])
        prediction = int(probability >= 0.5)
        profit = float(retention_profit(probability))

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            retention_profit=round(profit, 2),
            should_target=profit > 0,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def prepare_features(customer: CustomerData) -> pd.DataFrame:
    raw_data = customer.model_dump(by_alias=True)
    input_df = pd.DataFrame([raw_data])

    input_df["Service_count"] = (input_df[SERVICE_COLUMNS] != "No").sum(axis=1)
    input_df = input_df.drop(columns=SERVICE_COLUMNS)

    input_df["customer_loyalty"] = pd.cut(
        input_df["Tenure Months"],
        bins=[0, 1, 2, 10, 29, 72],
        labels=["new", "seeker", "floater", "loyal", "extra_loyal"],
        include_lowest=True,
    )
    input_df = input_df.drop(columns=["Tenure Months"])

    return input_df
