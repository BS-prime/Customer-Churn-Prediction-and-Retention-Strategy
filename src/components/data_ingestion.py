# import modules
import logging
import sys
from pathlib import Path
from exception import CustomException
from src.config import load_config

# import libraries
import pandas as pd

config = load_config()

# locate the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent


def data_ingestion(path: Path = config["data"]["path"]):
    """

    :param path:
    :return:
    """
    try:
        logging.info(f"Loading data from {path}")
        df: pd.DataFrame = pd.read_excel(ROOT_DIR / path)
        return df

    except Exception as e:
        raise CustomException(e, sys)