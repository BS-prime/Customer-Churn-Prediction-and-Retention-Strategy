# import modules
import logging
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

from exception import CustomException
from src.config import load_config

# import libraries
import pandas as pd

# initiate the config
config = load_config()

# locate the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent





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
        X_train, X_test, y_train, y_test = _train_test_split(X, y)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        raise CustomException(e, sys)
