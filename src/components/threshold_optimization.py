# Import modules
import json
from pathlib import Path
from typing import Any

# Import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Import functions
from src.config import load_config
from src.logger import logging

# locate root directory
ROOT_DIR = Path(__file__).resolve().parents[2]


def make_prediction(model: Any, x: pd.DataFrame) -> np.ndarray:
    """
    Helper function for making predictions using a trained model.
    :param model:
    :param x: pandas.DataFrame
    :return: predict probability
    """

    logging.info("Predicting...probabilities")

    return model.predict_proba(x)[:, 1]


def calculate_cost(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    fp_cost: float = 1.0,
    fn_cost: float = 10.0,
) -> float | int:
    """
    Helper function calculate the cost of making false positive and
    false negative error by the model
    :param y_true: pandas.Series
    :param y_pred: pandas.Series
    :param fp_cost: float
    :param fn_cost: float
    :return: float
    """

    # calculate the fp and fn cases
    _, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    logging.info("Calculating cost...")

    return fp_cost * fp + fn_cost * fn


def search_optimal_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    fp_cost: float = 1.0,
    fn_cost: float = 10.0,
) -> dict[str, float]:
    """
    Search the threshold that minimizes false-positive and false-negative cost.
    :param y_true: true target labels
    :param y_prob: predicted positive-class probabilities
    :param fp_cost: float
    :param fn_cost: float
    :return: threshold information
    """
    best_threshold = 0.5
    min_cost = float("inf")
    thresholds = np.linspace(0.01, 0.99, num=99)

    logging.info("Searching threshold...")

    for threshold in thresholds:
        # convert into integers
        y_pred = (np.asarray(y_prob) > threshold).astype(int)

        cost = calculate_cost(y_true, y_pred, fp_cost, fn_cost)

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

    return threshold_info


def save_threshold_info(
    threshold_info: dict[str, float],
    threshold_path: Path | None = None,
) -> Path:
    """
    Save threshold information to the configured threshold artifact path.
    :param threshold_info: threshold information
    :param threshold_path: optional output path
    :return: saved threshold path
    """
    logging.info("Saving thresholds...")

    # create threshold directory
    if threshold_path is None:
        config = load_config()
        threshold_path = ROOT_DIR / Path(config["output"]["threshold_path"])

    threshold_path.parent.mkdir(parents=True, exist_ok=True)

    with open(threshold_path, "w") as f:
        json.dump(threshold_info, f)

    logging.info(f"Threshold saved at: {threshold_path}")

    return threshold_path


def find_optimal_threshold(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    fp_cost: float = 1.0,
    fn_cost: float = 10.0,
) -> dict[str, float]:
    """
    Function to find optimal threshold to reduce cost and saves it in a json file
    :param fn_cost: float
    :param model: pickle object
    :param x_test: pandas.DataFrame
    :param y_test: pandas.Series
    :param fp_cost: float
    :return: dict
    """
    y_prob = make_prediction(x=x_test, model=model)
    threshold_info = search_optimal_threshold(
        y_true=y_test,
        y_prob=y_prob,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )
    save_threshold_info(threshold_info)

    return threshold_info
