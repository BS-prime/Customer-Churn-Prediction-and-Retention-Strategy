# Import modules
import json
import sys
from pathlib import Path

# Import libraries
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    balanced_accuracy_score
)

# Import functions
from src.config import load_config
from src.components.threshold_optimization import search_optimal_threshold
from src.exception import CustomException
from src.logger import logging

# locate the root directory
ROOT_DIR = Path(__file__).parents[2]


def evaluate_models(
        models: list,
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> tuple[object, dict, dict]:
    """
    Evaluate models performance
    :param models: list of models
    :param X_test: pd dataframe
    :param y_test: pd Series
    :return:
    """
    try:
        config_file = load_config()

        all_metrics = {}
        best_model = None
        best_threshold_info = {}
        best_cost = float("inf")

        for model in models:
            # extract the model name
            model_name = type(model).__name__
            logging.info(f"Evaluating model {model_name}")

            # calculate probability and make predictions
            probs = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            threshold_info = search_optimal_threshold(
                y_true=y_test,
                y_prob=probs,
            )   

            # Evaluation metrics
            auc = roc_auc_score(y_test, probs)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            balanced_accuracy_score_ = balanced_accuracy_score(y_test, y_pred)

            # put them into dictionary
            metrics = {
                "roc_auc": float(auc),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "class_balanced_accuracy": float(balanced_accuracy_score_),
                "best_threshold": threshold_info["best_threshold"],
                "min_cost": threshold_info["min_cost"],
                "cost_fp": threshold_info["cost_fp"],
                "cost_fn": threshold_info["cost_fn"],
            }

            # save the metrics
            all_metrics[model_name] = metrics

            # select best model using optimized business cost
            if threshold_info["min_cost"] < best_cost:
                best_cost = threshold_info["min_cost"]
                best_model = model
                best_threshold_info = threshold_info

            logging.info(f"Evaluated model: {model_name}")

        # create the directory if not present
        metrics_path = ROOT_DIR / Path(config_file["output"]["metrics_path"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        # Save all metrics to a single JSON file
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)

        logging.info(f"Metrics saved to: {metrics_path}")

        return best_model, all_metrics, best_threshold_info

    except Exception as e:
        raise CustomException(e, sys)
