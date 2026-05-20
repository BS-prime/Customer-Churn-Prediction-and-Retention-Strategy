# Import modules
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

# Import functions
from src.api.schemas import CustomerData, PredictionResponse
from src.pipeline.predict_pipeline import ChurnPredictor


@asynccontextmanager
async def lifespan(api: FastAPI):
    api.state.predictor = ChurnPredictor.from_artifacts()
    yield


app = FastAPI(
    title="Customer Churn Predictor 2026",
    description="Predict customer churn and retention target value",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "Customer Churn Predictor 2026 is up and running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    try:
        prediction = app.state.predictor.predict(customer.to_model_features())
        return PredictionResponse(**prediction)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed with exception {e}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
