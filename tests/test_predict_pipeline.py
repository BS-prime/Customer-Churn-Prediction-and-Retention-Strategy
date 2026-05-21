import pytest
import numpy as np

from src.pipeline.predict_pipeline import ChurnPredictor


class FakePreprocessor:
    feature_names_in_ = [
        "City",
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges",
        "CLTV",
        "Service_count",
        "customer_loyalty",
    ]

    def transform(self, input_df):
        assert input_df.columns.tolist() == self.feature_names_in_
        return input_df


class FakeModel:
    def predict_proba(self, processed_input):
        return np.array([[0.25, 0.75]])


def test_predictor_reorders_features_to_match_artifact_schema():
    predictor = ChurnPredictor(
        preprocessor=FakePreprocessor(),
        model=FakeModel(),
        threshold=0.5,
    )
    raw_features = {
        "Monthly Charges": 100,
        "Total Charges": 1000,
        "CLTV": 4226,
        "Service_count": 1,
        "City": "Anaheim",
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "customer_loyalty": "new",
    }

    prediction = predictor.predict(raw_features)

    assert prediction["churn_probability"] == 0.75
    assert prediction["will_churn"] is True


def test_predictor_rejects_missing_features():
    predictor = ChurnPredictor(
        preprocessor=FakePreprocessor(),
        model=FakeModel(),
        threshold=0.5,
    )

    with pytest.raises(ValueError, match="Missing required model features"):
        predictor.predict({"City": "Anaheim"})


def test_predictor_rejects_extra_features():
    predictor = ChurnPredictor(
        preprocessor=FakePreprocessor(),
        model=FakeModel(),
        threshold=0.5,
    )
    raw_features = {
        feature: "No" for feature in FakePreprocessor.feature_names_in_
    }
    raw_features.update({
        "Monthly Charges": 100,
        "Total Charges": 1000,
        "CLTV": 4226,
        "Service_count": 1,
        "Extra": "unexpected",
    })

    with pytest.raises(ValueError, match="Unexpected model features"):
        predictor.predict(raw_features)
