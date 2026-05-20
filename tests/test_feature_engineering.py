import pandas as pd

from src.components.feature_engineering import feature_engineering


def test_feature_engineering_creates_model_ready_features():
    df = pd.DataFrame(
        {
            "Churn Value": [0, 1],
            "Churn Label": ["No", "Yes"],
            "Churn Reason": ["N/A", "Competitor"],
            "Latitude": [1.0, 2.0],
            "Longitude": [3.0, 4.0],
            "Lat Long": ["1,3", "2,4"],
            "Zip Code": [90001, 90002],
            "Churn Score": [10, 80],
            "Phone Service": ["Yes", "No"],
            "Multiple Lines": ["No", "Yes"],
            "Internet Service": ["DSL", "No"],
            "Online Security": ["No", "Yes"],
            "Online Backup": ["Yes", "No"],
            "Device Protection": ["No", "Yes"],
            "Tech Support": ["Yes", "No"],
            "Streaming TV": ["No", "Yes"],
            "Streaming Movies": ["Yes", "No"],
            "Tenure Months": [1, 72],
            "Monthly Charges": [50.0, 90.0],
            "City": ["Los Angeles", "San Diego"],
        }
    )

    X, y = feature_engineering(df)

    assert y.tolist() == [0, 1]
    assert "Service_count" in X.columns
    assert X["Service_count"].tolist() == [5, 4]
    assert "customer_loyalty" in X.columns
    assert X["customer_loyalty"].astype(str).tolist() == ["new", "extra_loyal"]
    assert "Tenure Months" not in X.columns
    assert {
        "Churn Label",
        "Churn Reason",
        "Latitude",
        "Longitude",
        "Lat Long",
        "Zip Code",
        "Churn Score",
    }.isdisjoint(X.columns)
