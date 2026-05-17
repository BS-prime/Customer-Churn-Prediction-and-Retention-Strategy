# import modules
import joblib
from pathlib import Path

from src.config import load_config

# import libraries
import pandas as pd


def churn_probability(input_dict: dict) -> float:
    """
    predict churn probability
    :param input_dict:
    :return: float
    """

    # convert the input into an array
    input_array = pd.Dataframe(input_dict)

    # load the config file
    config = load_config()

    # locate the model directory
    ROOT_DIR = Path(__file__).parents[2]
    model_path = ROOT_DIR / Path(config["output"]["model_path"])

    # load the model
    model = joblib.load(model_path)

    # make prediction
    prob = model.predict_proba(input_array)[0, 1]

    return prob