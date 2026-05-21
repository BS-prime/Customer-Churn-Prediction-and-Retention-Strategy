import pytest
from pydantic import ValidationError

from src.ui.streamlit_app import build_customer_data


def test_build_customer_data_uses_shared_schema():
    customer = build_customer_data(
        {
            "MonthlyCharges": 100,
            "TotalCharges": 1000,
            "CLTV": 4226,
            "ServiceCount": 1,
            "City": "Los Angeles",
            "Gender": "Male",
            "SeniorCitizen": "No",
            "Partner": "No",
            "Dependents": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "CustomerLoyalty": "new",
        }
    )

    assert customer.to_model_features()["City"] == "Los Angeles"


def test_build_customer_data_rejects_invalid_values():
    with pytest.raises(ValidationError):
        build_customer_data(
            {
                "MonthlyCharges": -1,
                "TotalCharges": 1000,
                "CLTV": 4226,
                "ServiceCount": 1,
                "City": "Los Angeles",
                "Gender": "Male",
                "SeniorCitizen": "No",
                "Partner": "No",
                "Dependents": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "CustomerLoyalty": "new",
            }
        )
