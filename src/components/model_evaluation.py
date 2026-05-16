# import modules
import json
import sys
from pathlib import Path

# import libraries
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    balanced_accuracy_score
)

from src.config import load_config
from src.exception import CustomException
from src.logger import logging

# locate the root directory
ROOT_DIR = Path(__file__).parents[2]


def evaluate_models(
        models: list,
        X_test: pd.DataFrame,
        y_test: pd.Series
):
    """
    Evaluate models performance
    :param models: list of models
    :param X_test: pd dataframe
    :param y_test: pd Series
    :return:
    """
    try:
        config_file = load_config()

        # initiate variable
        all_metrics = {}
        best_model = None
        best_f1 = 0
        best_recall = 0

        for model in models:
            # extract the model name
            model_name = type(model).__name__
            logging.info(f"Evaluating model {model_name}")

            # calculate probability and make predictions
            probs = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # Evaluation metrics
            auc = roc_auc_score(y_test, probs)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            balanced_accuracy_score_ = balanced_accuracy_score(y_test, y_pred)

            # put them into dictionary
            metrics = {
                "roc_auc": float(auc),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "class_balanced_accuracy": float(balanced_accuracy_score_),
            }

            # save the metrics to
            all_metrics[model_name] = metrics

            # condition to check best model
            if recall > best_recall:
                if f1 > best_f1:
                    best_f1 = f1
                    best_recall = recall
                    best_model = model

            # create the directory if not present
            metrics_path = ROOT_DIR / Path(config_file["output"]["metrics_path"])
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            # Save all metrics to a single JSON file
            with open(metrics_path, "w") as f:
                json.dump(all_metrics, f, indent=4)

            logging.info(f"Metrics saved to: {metrics_path} and best model: {model_name}")

        return best_model, all_metrics

    except Exception as e:
        raise CustomException(e, sys)
