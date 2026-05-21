from pathlib import Path

import numpy as np
import pytest
import shap

from src.components import model_explainability
from src.components.model_explainability import (
    create_shap_explanation,
    sample_rows,
    select_positive_class_explanation,
)


class FakePreprocessor:
    def get_feature_names_out(self):
        return np.array(["num__Monthly Charges", "cat__Contract_Two year"])


class FakeModel:
    def predict_proba(self, X):
        return np.c_[1 - X[:, 0], X[:, 0]]


def test_sample_rows_limits_to_max_samples():
    X = np.arange(30).reshape(10, 3)

    sample = sample_rows(X, max_samples=4)

    assert sample.shape == (4, 3)
    assert sample.tolist() == X[:4].tolist()


def test_select_positive_class_explanation_for_binary_classifier():
    explanation = shap.Explanation(
        values=np.array(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ]
        ),
        base_values=np.array([[0.9, 1.0], [1.1, 1.2]]),
        data=np.array([[1, 2], [3, 4]]),
        feature_names=["a", "b"],
    )

    selected = select_positive_class_explanation(explanation)

    assert selected.values.tolist() == [[0.2, 0.4], [0.6, 0.8]]
    assert selected.base_values.tolist() == [1.0, 1.2]
    assert selected.feature_names == ["a", "b"]


def test_create_shap_explanation_rejects_feature_name_mismatch():
    X = np.array([[0.1, 0.2, 0.3]])

    with pytest.raises(Exception, match="Feature name count does not match"):
        create_shap_explanation(
            model=FakeModel(),
            X=X,
            preprocessor=FakePreprocessor(),
        )


def test_save_shap_summary_plot_writes_image(tmp_path, monkeypatch):
    output_path = tmp_path / "shap_summary.png"
    explanation = shap.Explanation(
        values=np.array([[0.2, 0.4]]),
        base_values=np.array([1.0]),
        data=np.array([[1, 2]]),
        feature_names=["a", "b"],
    )

    monkeypatch.setattr(
        model_explainability,
        "create_shap_explanation",
        lambda **kwargs: explanation,
    )
    monkeypatch.setattr(
        model_explainability.shap,
        "summary_plot",
        lambda *args, **kwargs: None,
    )

    saved_path = model_explainability.save_shap_summary_plot(
        model=FakeModel(),
        X=np.array([[0.1, 0.2]]),
        preprocessor=FakePreprocessor(),
        output_path=output_path,
    )

    assert saved_path == output_path
    assert Path(saved_path).exists()
