# import modules
import logging
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

from exception import CustomException
from src.config import load_config

# import libraries
import pandas as pd

config = load_config()

# locate the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent


def _input_output_split(df: pd.DataFrame):
    """
    Helper to split data into input and output features
    :param df:
    :return: the input and output dataframe
    """
    X = df.drop("Churn Value", axis=1)
    y = df["Churn Value"]
    return X, y


def _train_test_split(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = config["data"]["test_size"],
        random_state: int = config["data"]["random_state"]
):
    """
    Helper to perform train and test split
    :param data: pandas dataframe
    :param test_size: float
    :return: X_train, X_test, y_train, y_test
    """

    # train test data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state
                                                        )

    return X_train, X_test, y_train, y_test


def load_csv(path: Path = config["data"]["path"]):
    """
    load csv file and perform train_test_split
    :param path:
    :return: X_train, X_test, y_train, y_test
    """
    try:
        logging.info(f"Loading data from {path}")
        dataframe: pd.DataFrame = pd.read_excel(ROOT_DIR / path)

        # performing input output split
        X, y = _input_output_split(dataframe)
        logging.info(f"Shape of X: {X.shape}")
        logging.info(f"Shape of y: {y.shape}")

        logging.info("Performing train and test split")

        return _train_test_split(X,y)

    except Exception as e:
        raise CustomException(e, sys)
