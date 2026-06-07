# Import modules
import json
from pathlib import Path
from typing import Any

import dill

# Import libraries
import pandas as pd

# Import functions
from src.retention_strategy.retention import retention_profit

# ==============================================================================
# --- Prediction artifact paths ---
# ==============================================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

PREPROCESSOR_PATH = ROOT_DIR / "artifacts/preprocessor/preprocessor.pkl"
MODEL_PATH = ROOT_DIR / "models/best_model.pkl"
THRESHOLD_PATH = ROOT_DIR / "artifacts/threshold/threshold.json"

EXPECTED_MODEL_FEATURES = [
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


class ChurnPredictor:
    """
    Make customer churn prediction
    """

    def __init__(
        self,
        preprocessor: Any,
        model: Any,
        threshold: float,
        model_version: str = "1.0.0",
    ):
        self.preprocessor = preprocessor
        self.model = model
        self.threshold = threshold
        self.model_version = model_version

    @classmethod
    def load_artifacts(
        cls,
        preprocessor_path: Path = PREPROCESSOR_PATH,
        model_path: Path = MODEL_PATH,
        threshold_path: Path = THRESHOLD_PATH,
    ) -> "ChurnPredictor":
        """
        Load the saved preprocessor, model, and optimized threshold.
        """
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not threshold_path.exists():
            raise FileNotFoundError(f"Threshold not found: {threshold_path}")

        with open(preprocessor_path, "rb") as f:
            preprocessor = dill.load(f)

        with open(model_path, "rb") as f:
            model = dill.load(f)

        with open(threshold_path, "r") as f:
            threshold_data = json.load(f)

        return cls(
            preprocessor=preprocessor,
            model=model,
            threshold=threshold_data["best_threshold"],
        )

    def _expected_features(self) -> list[str]:
        """
        Helper to extract the feature names from the preprocessor.
        """
        artifact_features = getattr(self.preprocessor, "feature_names_in_", None)

        if artifact_features is None:
            return EXPECTED_MODEL_FEATURES

        return list(artifact_features)

    def _validate_features(self, raw_features: dict[str, Any]) -> dict[str, Any]:
        """
        Helper function to validate the input features
        """
        expected_features = self._expected_features()
        incoming_features = set(raw_features)
        missing_features = [
            feature for feature in expected_features if feature not in incoming_features
        ]
        extra_features = sorted(incoming_features - set(expected_features))

        if missing_features:
            raise ValueError(f"Missing required model features: {missing_features}")
        if extra_features:
            raise ValueError(f"Unexpected model features: {extra_features}")

        return {feature: raw_features[feature] for feature in expected_features}

    def predict(self, raw_features: dict[str, Any]) -> dict[str, float | bool | str]:
        """
        Score one customer and return churn and retention decision outputs.
        """
        validated_features = self._validate_features(raw_features)
        user_input = pd.DataFrame([validated_features])
        processed_input = self.preprocessor.transform(user_input)

        probability = self.model.predict_proba(processed_input)[0, 1]
        will_churn = bool(int(probability >= self.threshold))
        profit = retention_profit(prob=probability)

        return {
            "model_version": self.model_version,
            "churn_probability": round(float(probability), 4),
            "will_churn": will_churn,
            "profit": round(float(profit), 2),
            "should_target": bool(profit > 0),
        }
