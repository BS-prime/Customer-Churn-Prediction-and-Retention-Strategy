import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config import load_config
from logger import logging

# locate root directory
ROOT_DIR = Path(__file__).resolve().parents[2]


def _make_prediction(model: pickle, x: pd.DataFrame) -> pd.Series:
    """
    Helper function for making predictions using a trained model.
    :param model:
    :param x: pandas.DataFrame
    :return: predict probability
    """

    logging.info("Predicting...probabilities")

    return model.predict_proba(x)[0, 1]


def _calculate_cost(
        y_test: pd.Series,
        y_pred: pd.Series,
        fp_cost: float = 1.0,
        fn_cost: float = 10.0,
) -> float | int:
    """
    Helper function calculate the cost of making false positive and
    false negative error by the model
    :param y_test: pandas.Series
    :param y_pred: pandas.Series
    :param fp_cost: float     :param fn_cost:
    :return: float
    """

    # calculate the fp and fn cases
    _, fp, fn, _ = confusion_matrix(y_test, y_pred).ravel()

    logging.info("Calculating cost...")

    return fp_cost * fp + fn_cost * fn


def find_optimal_threshold(
        model: object,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        fp_cost: float = 1.0,
        fn_cost: float = 10.0,
) -> dict[str, float]:
    """
    Function to find optimal threshold to reduce cost.
    :param fn_cost: float
    :param model: pickle object
    :param x_test: pandas.DataFrame
    :param y_test: pandas.Series
    :param fp_cost: float
    :return: dict
    """

    # predict probability
    y_prob = _make_prediction(x=x_test, model=model)

    best_threshold = 0.5
    min_cost = float("inf")
    thresholds = np.linspace(0.01, 0.99, num=99)

    logging.info("Searching threshold...")

    for threshold in thresholds:

        # convert into integers
        y_pred = (y_prob > threshold).astype(int)

        cost = _calculate_cost(y_test, y_pred, fp_cost, fn_cost)

        # Threshold condition based on cost
        if cost < min_cost:
            min_cost = cost
            best_threshold = float(threshold)

    logging.info("Best threshold: " + str(best_threshold))

    threshold_info = {
        "best_threshold": best_threshold,
        "min_cost": float(min_cost),
        "cost_fp": fp_cost,
        "cost_fn": fn_cost,
    }

    logging.info("Saving thresholds...")

    # create threshold directory
    config = load_config()
    threshold_path = ROOT_DIR / Path(config["output"]["threshold_path"])
    threshold_path.parent.mkdir(parents=True, exist_ok=True)

    with open(threshold_path, "w") as f:
        json.dump(threshold_info, f)

    logging.info(f"Threshold saved at: {threshold_path}")

    return threshold_info
