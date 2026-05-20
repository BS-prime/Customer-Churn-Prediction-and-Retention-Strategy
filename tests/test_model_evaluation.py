import pandas as pd
import numpy as np

from src.components import model_evaluation


class HighCostModel:
    def predict_proba(self, X):
        return np.array([[0.1, 0.9], [0.8, 0.2], [0.8, 0.2], [0.2, 0.8]])

    def predict(self, X):
        return [1, 0, 0, 1]


class LowCostModel:
    def predict_proba(self, X):
        return np.array([[0.9, 0.1], [0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])

    def predict(self, X):
        return [0, 1, 1, 0]


def test_evaluate_models_selects_lowest_cost_model(tmp_path, monkeypatch):
    monkeypatch.setattr(model_evaluation, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(
        model_evaluation,
        "load_config",
        lambda: {"output": {"metrics_path": "metrics/metrics.json"}},
    )

    X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_test = pd.Series([0, 1, 1, 0])

    high_cost_model = HighCostModel()
    low_cost_model = LowCostModel()
    best_model, metrics, threshold_info = model_evaluation.evaluate_models(
        [high_cost_model, low_cost_model],
        X_test,
        y_test,
    )

    assert best_model is low_cost_model
    assert threshold_info["min_cost"] == 0.0
    assert metrics["LowCostModel"]["min_cost"] == 0.0
    assert (tmp_path / "metrics/metrics.json").exists()
