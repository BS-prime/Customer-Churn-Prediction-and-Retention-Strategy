import pytest
from pydantic import ValidationError

from src.api.schemas import CustomerData, MODEL_FEATURES, PredictionResponse


def test_customer_data_defaults_validate_and_map_to_model_features():
    customer = CustomerData()

    features = customer.to_model_features()

    assert list(features) == MODEL_FEATURES
    assert features["Monthly Charges"] == customer.MonthlyCharges
    assert features["Total Charges"] == customer.TotalCharges
    assert features["Service_count"] == customer.ServiceCount
    assert features["customer_loyalty"] == customer.CustomerLoyalty


def test_customer_data_accepts_any_non_empty_city():
    customer = CustomerData(City="Anaheim")

    assert customer.to_model_features()["City"] == "Anaheim"


def test_customer_data_rejects_invalid_category():
    with pytest.raises(ValidationError):
        CustomerData(Gender="Unknown")


def test_customer_data_rejects_empty_city():
    with pytest.raises(ValidationError):
        CustomerData(City="")


def test_prediction_response_contract():
    response = PredictionResponse(
        model_version="1.0.0",
        churn_probability=0.75,
        will_churn=True,
        profit=120.5,
        should_target=True,
    )

    assert response.model_dump() == {
        "model_version": "1.0.0",
        "churn_probability": 0.75,
        "will_churn": True,
        "profit": 120.5,
        "should_target": True,
    }
