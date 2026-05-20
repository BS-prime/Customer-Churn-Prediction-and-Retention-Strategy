# import modules
import logging
import sys
from pathlib import Path

# import libraries
import pandas as pd

from src.config import load_config
from src.exception import CustomException

# initiate the config
config = load_config()

# locate the root directory
ROOT_DIR = Path(__file__).resolve().parents[2]


def load_excel(path: Path = ROOT_DIR / config["data"]["path"]):
    """
    load csv file and perform train_test_split
    :param path:
    :return: X_train, X_test, y_train, y_test
    """
    try:
        logging.info(f"Loading data from {path}")

        dataframe: pd.DataFrame = pd.read_excel(ROOT_DIR / path)

        logging.info(f"Loaded dataframe: {dataframe.shape}")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)
